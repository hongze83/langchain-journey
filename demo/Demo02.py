# qwen_langgraph_demo.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import SystemMessage, HumanMessage

# 加载 .env 中的 API 密钥
load_dotenv()

# 初始化 Qwen 模型（通过 DashScope 的 OpenAI 兼容接口）
llm = ChatOpenAI(
    model="qwen3-max",  # 也可用 qwen-plus, qwen-turbo
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7,
)

# 定义节点函数：调用 Qwen
def call_qwen(state: MessagesState):
    # 调用 LLM，传入完整消息历史
    response = llm.invoke(state["messages"])
    # 返回新消息（AIMessage）
    return {"messages": [response]}

# 构建 LangGraph
builder = StateGraph(MessagesState)
builder.add_node("qwen_node", call_qwen)
builder.add_edge(START, "qwen_node")
builder.add_edge("qwen_node", END)

# 编译为可运行应用
graph = builder.compile()

# 测试调用
if __name__ == "__main__":
    system_prompt = "你是一个友好且鼓励性的教练，回答要简洁并给出入门建议。"
    user_input = "你好，我第一次使用lang graph使用千问开发agent，请鼓励一下我吧，并给我介绍下我如何更好的入门"

    result = graph.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
    })
    
    # 打印 AI 回复
    ai_message = result["messages"][-1]
    print("千问回答：")
    print(ai_message.content)