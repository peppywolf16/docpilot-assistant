import json
import re
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.retriever import smart_search
from app.generation.verifier import generate_verified_answer
from app.llm import generate


# Define the upgraded AgentState with conversation memory
class AgentState(TypedDict):
    question: str
    original_question: str
    chat_history: list[dict]       # List of {"role": "user"|"assistant", "content": "..."}
    passages: list[dict]
    relevance_score: float
    retry_count: int
    answer: str
    citations: list[dict]
    is_supported: bool


def condense_query_node(state: AgentState) -> dict:
    """Node 0: Uses conversation history to resolve pronouns ('it', 'that') into a standalone query."""
    history = state.get("chat_history", [])
    raw_q = state["question"]

    if not history:
        return {"original_question": raw_q}

    print("--- AGENT: CONDENSING QUERY WITH CONVERSATION MEMORY ---")
    formatted_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]])

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone technical question.
Do NOT answer the question. Only return the standalone rephrased question.

Chat History:
{formatted_history}

Follow-up Question: {raw_q}
Standalone Question:"""

    standalone_q = generate(prompt).strip().strip('"')
    print(f"--- AGENT: RESOLVED QUERY = '{standalone_q}' ---")
    return {"question": standalone_q, "original_question": raw_q}


def retrieve_node(state: AgentState) -> dict:
    """Node 1: Retrieves candidate passages using smart two-stage search."""
    print(f"--- AGENT: RETRIEVING FOR '{state['question']}' ---")
    passages = smart_search(query=state["question"], top_k=4)
    return {"passages": passages}


def grade_documents_node(state: AgentState) -> dict:
    """Node 2: Grades whether retrieved documents are relevant (LLM + Cross-Encoder logit check)."""
    print("--- AGENT: GRADING RETRIEVED DOCUMENTS ---")
    passages = state["passages"]
    if not passages:
        return {"relevance_score": 0.0}

    # 1. Cross-Encoder check
    top_rerank_score = passages[0].get("score", 0.0)
    if top_rerank_score < -2.0:
        print(f"--- AGENT: RERANKER SCORE LOW ({top_rerank_score}) -> FORCING REWRITE ---")
        return {"relevance_score": 0.2}

    # 2. Strict LLM relevance check
    context = "\n\n".join([f"[{p['number']}] {p['text']}" for p in passages])
    prompt = f"""You are a strict technical relevance grader.
Does the provided documentation context contain information to answer: "{state['question']}"?
Return ONLY a JSON object: {{"score": 0.0 to 1.0}}

Passages:
{context}
JSON Output:"""

    raw = generate(prompt).strip()
    cleaned = re.sub(r"^```json\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        score = float(data.get("score", 0.0))
    except Exception:
        score = 0.5

    print(f"--- AGENT: RELEVANCE SCORE = {score} ---")
    return {"relevance_score": score}


def decide_after_grade(state: AgentState) -> str:
    """Conditional Edge 1: Decide whether to generate or rewrite query."""
    score = state["relevance_score"]
    retries = state.get("retry_count", 0)

    if score < 0.4 and retries < 2:
        return "rewrite"
    return "generate"


def rewrite_query_node(state: AgentState) -> dict:
    """Node 3: Rewrites query with technical terms if retrieval was poor."""
    print("--- AGENT: REWRITING QUERY ---")
    prompt = f"Rewrite this query using specific technical terms to search FastAPI docs: '{state['question']}'"
    new_q = generate(prompt).strip().strip('"')
    print(f"--- AGENT: NEW QUERY = '{new_q}' ---")
    return {
        "question": new_q,
        "retry_count": state.get("retry_count", 0) + 1
    }


def generate_node(state: AgentState) -> dict:
    """Node 4: Generates verified answer and citations from passages."""
    print("--- AGENT: DRAFTING VERIFIED ANSWER ---")
    result = generate_verified_answer(
        question=state["question"],
        passages=state["passages"],
        rerank_threshold=-4.0
    )
    return {
        "answer": result["answer"],
        "citations": result["citations"]
    }


def check_support_node(state: AgentState) -> dict:
    """Node 5 (Self-RAG): Verifies that the drafted answer's claims are strictly grounded in passages."""
    print("--- AGENT: CHECKING ANSWER GROUNDEDNESS (SELF-RAG) ---")
    answer = state["answer"]
    passages = state["passages"]

    if "i don't have enough information" in answer.lower():
        # Clean refusal is inherently grounded
        return {"is_supported": True}

    context = "\n\n".join([f"[{p['number']}] {p['text']}" for p in passages])
    prompt = f"""You are a strict hallucination checker.
Are all factual claims in the answer strictly supported by the documentation passages below?
Return ONLY a JSON object: {{"supported": true/false}}

Passages:
{context}

Answer:
{answer}
JSON Output:"""

    raw = generate(prompt).strip()
    cleaned = re.sub(r"^```json\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        supported = bool(data.get("supported", True))
    except Exception:
        supported = True

    print(f"--- AGENT: ANSWER SUPPORTED = {supported} ---")
    return {"is_supported": supported}


# Build State Machine
workflow = StateGraph(AgentState)

workflow.add_node("condense", condense_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("check_support", check_support_node)

# Set entry point
workflow.set_entry_point("condense")

# Graph Connections
workflow.add_edge("condense", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_after_grade, {"rewrite": "rewrite", "generate": "generate"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", "check_support")
workflow.add_edge("check_support", END)

# In-memory checkpointer for multi-turn thread memory
memory = MemorySaver()
rag_agent = workflow.compile(checkpointer=memory)


def run_agent(question: str, chat_history: list[dict] | None = None, thread_id: str = "default_user") -> dict:
    """Executes the agent with conversation memory and checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "question": question,
        "original_question": question,
        "chat_history": chat_history or [],
        "passages": [],
        "relevance_score": 0.0,
        "retry_count": 0,
        "answer": "",
        "citations": [],
        "is_supported": True,
    }
    final_state = rag_agent.invoke(initial_state, config=config)
    return {
        "answer": final_state.get("answer"),
        "citations": final_state.get("citations"),
        "sources": final_state.get("passages"),
        "is_supported": final_state.get("is_supported"),
    }


def export_mermaid_diagram() -> str:
    """Returns the visual Mermaid diagram syntax of the state machine."""
    try:
        return rag_agent.get_graph().draw_mermaid()
    except Exception:
        return "graph TD;\n  condense-->retrieve-->grade;\n  grade-->|poor|rewrite-->retrieve;\n  grade-->|good|generate-->check_support-->END;"