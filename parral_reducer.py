import os
import re
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

def merge_score_dict(existing: dict, newupdate: dict) -> dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}

class State(TypedDict):
    raw_input: str
    safety_score: Annotated[dict[str, int], merge_score_dict]

def extract_score(text: str) -> int:
    """LLM ka response chahe JSON ho, 'key: value' ho, ya plain number—
    ismein se pehla integer (0-100) nikaal ke return karta hai."""
    text = text.strip()
    # sabse pehle pure integer try karo
    try:
        return int(text)
    except ValueError:
        pass
    # warna text mein jo bhi pehla number mile wo utha lo
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    # kuch bhi na mile to safe default
    return 0

# create the nodes
def toxicity_node(state: State) -> dict:
    print("Analyzing toxicity and hate speech of the input text...")
    prompt = (
        "Analyze the following text for toxicity and hate speech. "
        "Respond with ONLY a single integer from 0 to 100 (0 = no toxicity, "
        "100 = extreme toxicity). Do not include any words, JSON, or explanation—"
        "just the number."
        f"\n\nRaw Input:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    score = extract_score(response.content)
    return {"safety_score": {"toxicity_score": score}}

def copyright_node(state: State) -> dict:
    print("Analyzing copyright infringement of the input text...")
    prompt = (
        "Analyze the following text for potential copyright infringement. "
        "Respond with ONLY a single integer from 0 to 100 (0 = no copyright "
        "issues, 100 = high risk). Do not include any words, JSON, or "
        "explanation—just the number."
        f"\n\nRaw Input:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    score = extract_score(response.content)
    return {"safety_score": {"copyright_score": score}}

def culture_node(state: State) -> dict:
    print("Analyzing cultural sensitivity of the input text...")
    prompt = (
        "Analyze the following text for cultural sensitivity. "
        "Respond with ONLY a single integer from 0 to 100 (0 = no cultural "
        "insensitivity, 100 = high risk). Do not include any words, JSON, or "
        "explanation—just the number."
        f"\n\nRaw Input:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    score = extract_score(response.content)
    return {"safety_score": {"cultural_score": score}}

builder = StateGraph(State)

builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_node", copyright_node)
builder.add_node("culture_node", culture_node)

builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "culture_node")

builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_node", END)
builder.add_edge("culture_node", END)

app = builder.compile()

sample_input = """
I hate people from that country, they are all garbage and should be banned.
This is a direct copy paste from someone else's copyrighted novel word for word.
"""

initial_state = {"raw_input": sample_input, "safety_score": {}}

final_state = app.invoke(initial_state)
print("Final State:", final_state["safety_score"])