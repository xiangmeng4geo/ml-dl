from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SerpAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

llm = OllamaLLM(model="qwen2.5:14b", base_url="http://192.168.3.101:11434")

prompt = hub.pull("hwchase17/react")

params = {
    "gl": "cn",
    "hl": "zh-cn",
}
search = SerpAPIWrapper(params=params)
python_repl = PythonREPL()

# You can create the tool to pass to an agent
tools = [Tool(
    name="python_repl",
    description="一个 Python shell. 用于执行python代码。",
    func=python_repl.run,
), Tool(
    name="search",
    description="一个搜索工具，当大模型没有相关知识时，可以使用这个工具进行搜索。",
    func=search.run,
)]

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
res = agent_executor.invoke({"input": "学习LLM Agent比较好的书有哪些", "max_tokens": 100})
print(res)

