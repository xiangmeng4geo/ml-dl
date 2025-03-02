import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 文档存储目录
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# 索引存储目录
INDEX_DIR = os.path.join(ROOT_DIR, "index")

# Ollama设置
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama2"  # 或其他已在Ollama中安装的模型

# 创建必要的目录
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
