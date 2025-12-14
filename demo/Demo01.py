from langgraph.graph import StateGraph, MessagesState, START, END
import json

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

def message_to_dict(msg):
    return {"role": msg.type, "content": msg.content}

# 构建图
graph = StateGraph(MessagesState)
graph.add_node("mock_llm", mock_llm)  # ⚠️ 注意：这里要指定节点名！
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
app = graph.compile()

# 调用并打印结果
result = app.invoke({"messages": [{"role": "user", "content": "hi!"}]})
messages_as_dicts = [message_to_dict(m) for m in result["messages"]]

# 假设 messages_as_dicts 是你的消息列表
print(json.dumps(messages_as_dicts, indent=2, ensure_ascii=False))