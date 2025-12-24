import uuid
from typing import Optional
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    name: Optional[str]
    age: Optional[str]

def ask_name(state: State):
    name = interrupt("What's your name?")
    print(f"[ask_name] Got: {name}")
    return {"name": name}

def ask_age(state: State):
    age = interrupt("What's your age?")
    print(f"[ask_age] Got: {age}")
    return {"age": age}

builder = StateGraph(State)
builder.add_node("ask_name", ask_name)
builder.add_node("ask_age", ask_age)
builder.add_edge(START, "ask_name")
builder.add_edge("ask_name", "ask_age")
builder.add_edge("ask_age", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

for chunk in graph.stream({"name": None, "age": None}, config):
    print("→", chunk)

for chunk in graph.stream(Command(resume="Alice"), config):
    print("→", chunk)

for chunk in graph.stream(Command(resume="30"), config):
    print("→", chunk)