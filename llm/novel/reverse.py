from langchain_ollama import OllamaLLM
from langchain import hub

from langchain_community.utilities import SerpAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

from langchain.agents import create_react_agent, AgentExecutor

import json


prompt = hub.pull("hwchase17/react")
llm = OllamaLLM(model="qwen2.5:14b", base_url="http://192.168.3.101:11434")


params = {
    "gl": "cn",
    "hl": "zh-cn",
}
search = SerpAPIWrapper(params=params)
python_repl = PythonREPL()

tools = [
    Tool(
        name="python_repl",
        description="一个 Python shell. 用于执行python代码。",
        func=python_repl.run,
    ),
    Tool(
        name="search",
        description="一个搜索工具，当大模型没有相关知识时，可以使用这个工具进行搜索。",
        func=search.run,
    ),
]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def analyze_book(book_text):
    input_data = {
        "input": f"请分析以下小说的背景、故事主线和角色：\n{book_text}",
        "max_tokens": 1000
    }
    result = agent_executor.invoke(input_data)
    return result


def output_to_json(analysis_result, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=4)

# Example usage
book_text = "这里是小说的文本内容..."
analysis_result = analyze_book(book_text)
output_to_json(analysis_result, 'analysis_result.json')