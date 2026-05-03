"""
Pydantic schemas for the structured-output demo.

The `Field(description=...)` text is important: Pydantic puts it into the
generated JSON schema, which we then inject into the prompt. So the model
literally reads these descriptions when deciding what to put in each field.
That's a free, zero-token-cost way to align the model with your intent —
write descriptions for the model, not just for your IDE tooltip.
"""

from pydantic import BaseModel, Field


class FirstSection(BaseModel):
    origin_of_word: str = Field(
        ...,
        description=(
            "Etymology and historical origin of the word. 2-4 sentences. "
            "Mention the source language and approximate era when known."
        ),
    )


class SecondSection(BaseModel):
    essay: str = Field(
        ...,
        description=(
            "A single, well-structured essay of approximately 200 words on "
            "the word. Plain prose, no headings, no bullet points."
        ),
    )


class ThirdSection(BaseModel):
    word_usage: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description=(
            "3 to 5 example sentences demonstrating natural use of the word "
            "in different contexts. Each sentence should stand alone."
        ),
    )


class WordAnalysis(BaseModel):
    """Top-level structured response for a single word."""

    first: FirstSection
    second: SecondSection
    third: ThirdSection
