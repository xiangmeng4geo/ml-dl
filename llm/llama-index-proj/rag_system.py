from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import Ollama
from config import DOCS_DIR, INDEX_DIR, OLLAMA_BASE_URL, MODEL_NAME
import os

class RAGSystem:
    def __init__(self):
        # 初始化Ollama LLM
        self.llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
        
        # 创建ServiceContext
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            chunk_size=512,
            chunk_overlap=128
        )

        # 加载或创建索引
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        # 如果已有索引文件，直接加载
        index_file = os.path.join(INDEX_DIR, "index.json")
        if os.path.exists(index_file):
            return VectorStoreIndex.load_from_disk(
                index_file,
                service_context=self.service_context
            )
        
        # 否则创建新索引
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        # 保存索引
        index.storage_context.persist(persist_dir=index_file)
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
