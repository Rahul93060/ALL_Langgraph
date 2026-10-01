#so we have to creating a graph
#and the first thing you create a state

# 1). typed dict
from typing import TypedDict

class State(TypedDict):
    topic:str
    summary:str
    score:int 

#2) pydantic approch 
#it is good for the data validation
#runtime
from pydantic import BaseModel,field_validator

class StateModel(BaseModel):
    topic:str
    summary:str
    score:int

    @field_validator
    def score_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('score must be positive')

#3) python dataclass approch
#standard python dataclass but it is use rarely 

from dataclasses import dataclass,field

class state:
    topic:str
    summary:str
    msg:list =field(default_factory=list)

    
#4) from langraph
from langgraph.graph import MessagesState

class State(MessagesState):
    #msg field is already defined in the MessagesState class  just add extra fields
    usr_name:str
    language:str
    