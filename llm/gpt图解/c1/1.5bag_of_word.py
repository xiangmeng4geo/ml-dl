# bag of word词袋模型，判断文本相似度

import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager

# List all available fonts
for font in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
    print(font)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 示例文本
documents = [
    "我爱编程",
    "编程是一个很有趣的事情",
    "我喜欢学习新知识",
    "数据科学是未来的趋势",
    "机器学习改变世界",
    "深度学习是人工智能的核心",
    "自然语言处理让机器理解人类语言",
    "计算机视觉让机器看懂世界",
    "强化学习让机器学会决策",
    "大数据分析带来商业价值"
]


# 使用jieba分词
def jieba_tokenizer(text):
    return jieba.lcut(text)


# 创建词袋模型
vectorizer = CountVectorizer(tokenizer=jieba_tokenizer)
X = vectorizer.fit_transform(documents)

# 打印词袋模型的特征名称
print("Feature names:", vectorizer.get_feature_names_out())

# 打印词袋模型的稀疏矩阵
print("Bag of Words matrix:\n", X.toarray())


# 计算文本相似度（余弦相似度）使用numpy
def cosine_similarity_numpy(matrix):
    norm = np.linalg.norm(matrix, axis=1)
    similarity = np.dot(matrix, matrix.T) / (norm[:, None] * norm[None, :])
    return similarity


similarity_matrix_numpy = cosine_similarity_numpy(X.toarray())
print("Cosine similarity matrix (numpy):\n", similarity_matrix_numpy)

# 计算文本相似度（余弦相似度）使用scikit-learn
similarity_matrix_sklearn = cosine_similarity(X)
print("Cosine similarity matrix (scikit-learn):\n", similarity_matrix_sklearn)

# 可视化相似度矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix_numpy, annot=True, cmap='coolwarm', xticklabels=documents, yticklabels=documents)
plt.title('Cosine Similarity Matrix')
plt.show()
