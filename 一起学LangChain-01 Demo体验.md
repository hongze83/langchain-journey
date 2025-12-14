作为一名开发者，在大模型这么热的今天，决定不能落下，开始学起来，有兴趣的朋友可以一起，今天在[langchain官网](https://docs.langchain.com/oss/python/langgraph/overview)，大致看了下，决定先来个例子感受一下，然后在学习概念，准备直接拿官方的计算器Agent来练手！官方例子代码如下：<https://docs.langchain.com/oss/python/langgraph/quickstart>。我把大模型替换成了千问，其他都没变。

## 环境准备

> python：3.12 编译器：PyCharm python依赖包：langchain、langgraph、langchain\_community、dashscope、dotenv

python依赖包通过PyCharm->Preferences->Project->Python Interpreter添加

## 完整代码

    # Step 1: 定义工具和模型
    import os
    from langchain.tools import tool
    from langchain_community.chat_models import ChatTongyi
    from dotenv import load_dotenv

    load_dotenv()

    # 初始化 Qwen 模型（通过 DashScope 的 OpenAI 兼容接口）
    model = ChatTongyi(
        model="qwen-max",
        temperature=0,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )


    # 定义工具
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a * b


    @tool
    def add(a: int, b: int) -> int:
        """Adds `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a + b


    @tool
    def divide(a: int, b: int) -> float:
        """Divide `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a / b


    # 模型绑定工具
    tools = [add, multiply, divide]
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    # Step 2: 定义状态

    from langchain.messages import AnyMessage
    from typing_extensions import TypedDict, Annotated
    import operator


    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int

    # Step 3: 定义模型Node
    from langchain.messages import SystemMessage
    import json

    def llm_call(state: dict):
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }


    # Step 4: 定义工具Node

    from langchain.messages import ToolMessage

    def tool_node(state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    # Step 5: 定义逻辑决定是否结束，还是执行工具调用

    from typing import Literal
    from langgraph.graph import StateGraph, START, END


    # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END

    # Step 6: Build agent

    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # 添加Node
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # 编译 the agent
    agent = agent_builder.compile()

    # Show the agent
    # print(agent.get_graph(xray=True).draw_mermaid())

    # Invoke
    from langchain.messages import HumanMessage
    from langchain_core.runnables.config import RunnableConfig

    messages = [HumanMessage(content="Add 3 and 4.")]
    messages = agent.invoke(
        {"messages": messages},
        config=RunnableConfig(run_name="Arithmetic Agent - Add 3+4")
    )
    for m in messages["messages"]:
        m.pretty_print()

## 替换为千问大模型

上面代码，大模型部分已经被替换成了千问的模型，下面是替换过程

安装必要包：

    pip install langchain-core langchain-community dashscope

获取 DashScope API Key：

*   访问 [阿里云 DashScope 控制台](https://zhuanlan.zhihu.com/p/1983198808226694927/edit)
*   获取 API Key（如 sk-xxxxxx）

设置环境变量（推荐）

用户目录下.env文件，增加

    DASHSCOPE_API_KEY= 你的千问模型api key

替换模型部分代码

    from dotenv import load_dotenv
    load_dotenv()# 加载环境变量
    from langchain_community.chat_models import ChatTongyi

    model = ChatTongyi(
        model="qwen-max",
        temperature=0,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

## 执行流程分析

从打印日志可以看出：请求了两次大模型：

第一次调用：

*   发送消息：\[System, Human("Add 3 and 4.")]
*   返回消息：tool\_calls = \[{"name": "add", ...}]

第二次调用：

*   发送消息：\[System, Human("Add 3 and 4."), AI(tool\_calls), Tool("7")]
*   返回消息：content = "The result is 7."

我们想办法看看整个执行过程到底是如何进行的

### 方法一：打印成mermaid格式

被注释掉的代码 # print(agent.get\_graph(xray=True).draw\_mermaid()) 可以打印mermaid格式的流程图，在[https://mermaid.live](https://mermaid.live/)中可以生成执行流程图

    graph TD;
    	__start__([<p>__start__</p>]):::first
    	llm_call(llm_call)
    	tool_node(tool_node)
    	__end__([<p>__end__</p>]):::last
    	__start__ --> llm_call;
    	llm_call -.-> __end__;
    	llm_call -.-> tool_node;
    	tool_node --> llm_call;

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2e0ed1e19a2849c498bf7d2b177b1cbe~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5ram5rO95ou-5YWJ:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMTk2MzA1NzE5MjkyMjE3MCJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1766301540&x-orig-sign=G2ht3a4iUwGrhEAdjMI1PddlX0w%3D)

添加图片注释，不超过 140 字（可选）

### 方法二：接入LangSmith

可以接入langchain官网的LangSmith，接入非常简单。

✅ 第一步：注册并获取 LangSmith 凭据

1.  访问 <https://smith.langchain.com/>
2.  用 GitHub / Google 账号登录
3.  进入 Settings → API Keys
4.  创建一个新 API Key（或复制已有）
5.  同时记下你的 Project Name（默认是 default）

✅ 第二步：安装 LangSmith SDK

    bash编辑


    pip install langsmith

> 注意：langchain >= 0.1.0 已内置对 LangSmith 的支持，无需额外安装 langchain-core。

✅ 第三步：设置环境变量

在项目根目录创建 .env 文件（如果你还没做）：

    env编辑

    # DashScope API（你已配置）
    DASHSCOPE_API_KEY=sk-xxxxxx

    # 👇 新增 LangSmith 配置
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=lsk_xxxxxx  # 从 LangSmith 控制台复制
    LANGCHAIN_PROJECT=default      # 或你自定义的项目名

> 🔑 关键变量说明：

*   LANGCHAIN\_TRACING\_V2=true：启用追踪
*   LANGCHAIN\_API\_KEY：你的 LangSmith API Key
*   LANGCHAIN\_PROJECT：数据归集到哪个项目（可选，默认 default）

然后在代码开头加载：

    python编辑


    from dotenv import load_dotenv
    load_dotenv()  # 确保这行在最前面

✅ 第四步：为你的 Agent 添加唯一 traceable 名称（可选但推荐）

LangSmith 会自动追踪所有 LangChain 调用，但你可以给 agent.invoke 加个名字便于识别：

    python编辑

    result = agent.invoke(
        {"messages": [HumanMessage(content="Add 3 and 4.")]},
        config={"run_name": "Arithmetic Agent"}
    )

✅ 第五步：运行你的脚本

只要设置了环境变量，所有 LangChain 操作（LLM 调用、工具调用、节点执行）都会自动上报到 LangSmith。

✅ 第六步：在 LangSmith 查看可视化流程图

1.  打开 <https://smith.langchain.com/>
2.  进入 Datasets → Traces（或直接点左侧 "Traces"）
3.  找到你刚运行的 trace（按时间排序）
4.  点击进入，你会看到：

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6dffba83ccb246f6b02043acf014cfc0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5ram5rO95ou-5YWJ:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMTk2MzA1NzE5MjkyMjE3MCJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1766301540&x-orig-sign=WCN6LkUn3sw6UZ6VR5KYYO6DbVg%3D)

