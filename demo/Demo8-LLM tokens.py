import os
from dataclasses import dataclass

from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
load_dotenv()

@dataclass
class MyState:
    topic: str
    joke: str = ""



# 初始化 Qwen 模型（通过 DashScope 的 OpenAI 兼容接口）
model = ChatTongyi(
    model="qwen-max",
    streaming=True,
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)


def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    # Note that message events are emitted even when the LLM is run using .invoke rather than .stream
    model_response = model.invoke(
        [
            {"role": "user", "content": f"帮我介绍一下他的生平 {state.topic}"}
        ]
    )
    return {"joke": model_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# The "messages" stream mode returns an iterator of tuples (message_chunk, metadata)
# where message_chunk is the token streamed by the LLM and metadata is a dictionary
# with information about the graph node where the LLM was called and other information
for message_chunk, metadata in graph.stream(
    {"topic": "刘德华"},
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)