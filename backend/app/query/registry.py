from app.query.base import QueryTransform
from app.query.multi_query import MultiQueryTransform
from app.query.hyde import HyDETransform
from app.query.step_back import StepBackTransform

TRANSFORMS = {
    "none": None,
    "multi_query": MultiQueryTransform,
    "hyde": HyDETransform,
    "step_back": StepBackTransform,
}


def get_query_transform(name: str = "none") -> QueryTransform | None:
    """Factory to get the query transform strategy by name."""
    cls = TRANSFORMS.get(name.lower())
    return cls() if cls else None