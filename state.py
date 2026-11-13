# how to make a state in langgraph 
import os 
from typing import TypedDict

# =================================================================
# 1. TypedDict (Most common in LangGraph)
# =================================================================
# TypedDict is used to define the expected structure and data types of a dictionary.
# In simple words, it tells Python type checkers:
# - Which keys should exist in a dictionary.
# - What type of value each key should contain.

class DictState(TypedDict):
    topic: str
    summary: str
    score: int


# =================================================================
# 2. Pydantic BaseModel (Good for runtime type validation)
# =================================================================
from pydantic import BaseModel, field_validator

class PydanticState(BaseModel):
    topic: str
    score: int
    summary: str = ""

    @field_validator("score")
    @classmethod
    def score_positive(cls, value):
        if value < 0:
            raise ValueError("Score must be >= 0")
        return value


# =================================================================
# 3. Python Dataclasses (Used rarely)
# =================================================================
from dataclasses import dataclass, field

@dataclass
class DataclassState:
    topic: str = ""
    summary: str = ""
    message: list = field(default_factory=list)


# =================================================================
# 4. Built-in MessagesState (Standard for chatbot/conversational graphs)
# =================================================================
from langgraph.graph import MessagesState
# MessagesState automatically has a "messages" key with an add_messages reducer
