from llama_index.core import StorageContext, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage
from llama_index.llms.ollama import Ollama

from config import DOCS_DIR, INDEX_DIR, OLLAMA_BASE_URL, MODEL_NAME
import os


class RAGSystem:
    def __init__(self):
        # 初始化Ollama LLM
        self.llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

        # 加载或创建索引
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        # 如果已有索引文件，直接加载
        if os.path.exists(INDEX_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            return load_index_from_storage(storage_context)

        # 否则创建新索引
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
        )
        # 保存索引
        index.storage_context.persist(persist_dir=INDEX_DIR)
        return index

    def query(self, question: str) -> str:
        query_engine = self.index.as_query_engine()
        response = query_engine.query(question)
        return str(response)

    def add_document(self, file_path: str):
        """添加新文档到知识库"""
        new_docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        self.index.insert_nodes(new_docs)
        # 更新保存的索引
        self.index.storage_context.persist(
            persist_dir=os.path.join(INDEX_DIR, "index.json")
        )
