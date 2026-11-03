from app.store_qdrant import (
    add_document,
    delete_document,
    list_documents,
    client,
    COLLECTION_NAME,
)
from app.retriever import hybrid_search as search

class _ChunksProxy:
    def __len__(self) -> int:
        try:
            return client.count(collection_name=COLLECTION_NAME, exact=True).count
        except Exception:
            return 0

    def __bool__(self) -> bool:
        return len(self) > 0

CHUNKS = _ChunksProxy()