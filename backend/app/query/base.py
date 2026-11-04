from abc import ABC, abstractmethod


class QueryTransform(ABC):
    """Abstract Base Class for all query transformation strategies."""

    @abstractmethod
    def transform(self, query: str) -> list[str]:
        """
        Takes an input query string and returns a list of queries to search.
        For HyDE: returns [hypothetical_passage].
        For Multi-Query: returns [original_query, rephrase_1, rephrase_2, ...].
        For Step-Back: returns [original_query, broader_query].
        """
        pass