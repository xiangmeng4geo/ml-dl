from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "qwen2.5:32b",
    temperature = 0.8,
    num_predict = 256,
    base_url="http://192.168.3.100:11434"
)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to Chinese."),
    ("human", "I love programming."),
]
llm.invoke(messages)