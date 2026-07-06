from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, field_validator

class GraphTriplet(BaseModel):
    head: str = Field(..., description="Subject entity (e.g., BERT)")
    relation: str = Field(..., description="Relationship verb (e.g., IMPLEMENTS)")
    tail: str = Field(..., description="Object entity (e.g., Attention Mechanism)")
    confidence: float = Field(default=1.0, ge=0, le=1.0)

    @field_validator("head", "relation", "tail")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.replace("\n", " ").strip()

class TripletList(BaseModel):
    triplets: List[GraphTriplet] = Field(default_factory=list)
