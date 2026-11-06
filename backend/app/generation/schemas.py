from pydantic import BaseModel, Field


class Citation(BaseModel):
    passage_number: int = Field(description="The number of the passage cited, e.g. 1, 2")
    exact_quote: str = Field(description="An exact, word-for-word quote from that passage supporting the statement")


class GroundedResponse(BaseModel):
    answerable: bool = Field(description="False if the passages do not contain enough information to answer")
    answer: str = Field(description="The factual answer based strictly on the passages, or a clear refusal")
    citations: list[Citation] = Field(default_factory=list, description="List of exact quotes and their passage numbers")