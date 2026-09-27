# src/agent/agent.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# 新代码
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.agent.tools import search_pdf

load_dotenv()

# 1. 用 Qwen 作为 Agent 的大脑
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
)

# 2. 定义 Agent 的提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手。当用户询问 PDF 文档（强化学习/MDP/马尔可夫过程）相关内容时，"
               "必须调用 search_pdf 工具；闲聊类问题直接回答即可。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 3. 把工具挂到 Agent 上
tools = [search_pdf]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    print("\n=== 测试 1：需要查 PDF 的问题 ===")
    r1 = executor.invoke({"input": "什么是马尔可夫决策过程？"})
    print("回答：", r1["output"])

    print("\n=== 测试 2：闲聊（不该查 PDF） ===")
    r2 = executor.invoke({"input": "你好，今天天气怎么样？"})
    print("回答：", r2["output"])