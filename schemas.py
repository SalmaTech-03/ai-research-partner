# schemas.py

"""
Pydantic schemas used throughout the AI Research Partner.

Responsibilities
----------------
1. Validate extracted knowledge graph triplets
2. Ensure clean entity names
3. Prevent malformed LLM outputs
4. Maintain compatibility with Pydantic v2
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GraphTriplet(BaseModel):
    """
    Represents a single knowledge graph triplet.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    head: str = Field(
        ...,
        min_length=1,
        description="Head (subject) entity."
    )

    relation: str = Field(
        ...,
        min_length=1,
        description="Relationship between entities."
    )

    tail: str = Field(
        ...,
        min_length=1,
        description="Tail (object) entity."
    )

    @field_validator("head", "relation", "tail")
    @classmethod
    def clean_text(cls, value: str) -> str:
        """
        Normalize whitespace and remove line breaks.
        """
        return (
            value.replace("\n", " ")
                 .replace("\r", " ")
                 .strip()
        )


class TripletList(BaseModel):
    """
    Collection of extracted triplets.
    """

    model_config = ConfigDict(
        extra="ignore"
    )

    triplets: List[GraphTriplet] = Field(
        default_factory=list,
        description="List of extracted graph triplets."
    )
