# schemas.py

from pydantic import BaseModel, Field
from typing import List, Tuple

# Pydantic model for a single graph triplet
class GraphTriplet(BaseModel):
    """A single triplet representing a relationship in the knowledge graph."""
    head: str = Field(..., description="The subject or head entity of the triplet.")
    relation: str = Field(..., description="The relationship connecting the head and tail entities.")
    tail: str = Field(..., description="The object or tail entity of the triplet.")

# Pydantic model for the list of triplets
class TripletList(BaseModel):
    """A list of triplets extracted from a text chunk."""
    triplets: List[GraphTriplet]