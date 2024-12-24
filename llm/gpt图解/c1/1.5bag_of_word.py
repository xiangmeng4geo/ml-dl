# bag of word词袋模型，判断文本相似度

import jieba
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from math import log

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



class CountVectorizer:
    def __init__(self, tokenizer=jieba.lcut):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}
        self.feature_names_ = []

    def fit_transform(self, documents):
        tokenized_documents = [self.tokenizer(doc) for doc in documents]
        self._build_vocabulary(tokenized_documents)
        return self._transform(tokenized_documents)

    def _build_vocabulary(self, tokenized_documents):
        """
        构建词汇表
        :param tokenized_documents:
        :return:
        """
        vocab = set()
        for doc in tokenized_documents:
            vocab.update(doc)
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(vocab))}
        self.feature_names_ = sorted(vocab)

    def _transform(self, tokenized_documents):
        """
        将文档转换为词袋矩阵
        :param tokenized_documents:
        :return:
        """
        rows = len(tokenized_documents)
        cols = len(self.vocabulary_)
        matrix = np.zeros((rows, cols), dtype=int)
        for row, doc in enumerate(tokenized_documents):
            for word in doc:
                if word in self.vocabulary_:
                    col = self.vocabulary_[word]
                    matrix[row, col] += 1
        return matrix

    def get_feature_names_out(self):
        return self.feature_names_

class TfidfVectorizer(CountVectorizer):
    def fit_transform(self, documents):
        count_matrix = super().fit_transform(documents)
        tf_matrix = self._compute_tf(count_matrix)
        idf_vector = self._compute_idf(count_matrix)
        return tf_matrix * idf_vector

    def _compute_tf(self, count_matrix):
        tf_matrix = count_matrix.astype(float)
        for row in range(tf_matrix.shape[0]):
            row_sum = np.sum(tf_matrix[row])
            if row_sum > 0:
                tf_matrix[row] /= row_sum
        return tf_matrix

    def _compute_idf(self, count_matrix):
        doc_count = count_matrix.shape[0]
        idf_vector = np.zeros(count_matrix.shape[1])

        for col in range(count_matrix.shape[1]):
            doc_freq = np.sum(count_matrix[:, col] > 0)
            idf_vector[col] = log((doc_count + 1) / (doc_freq + 1)) + 1
        return idf_vector


# 创建词袋模型
vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
# vectorizer = CountVectorizer(tokenizer=jieba_tokenizer)
X = vectorizer.fit_transform(documents)

# 打印词袋模型的特征名称
print("Feature names:", vectorizer.get_feature_names_out())

# 打印词袋模型的稀疏矩阵
print("Bag of Words matrix:\n", X)


# 计算文本相似度（余弦相似度）使用numpy
def cosine_similarity_numpy(matrix):
    norm = np.linalg.norm(matrix, axis=1)
    similarity = np.dot(matrix, matrix.T) / (norm[:, None] * norm[None, :])
    return similarity


similarity_matrix_numpy = cosine_similarity_numpy(X)
print("Cosine similarity matrix (numpy):\n", similarity_matrix_numpy)

# 计算文本相似度（余弦相似度）使用scikit-learn
similarity_matrix_sklearn = cosine_similarity(X)
print("Cosine similarity matrix (scikit-learn):\n", similarity_matrix_sklearn)

# 可视化相似度矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix_numpy, annot=True, cmap='coolwarm', xticklabels=documents, yticklabels=documents)
plt.title('Cosine Similarity Matrix')
plt.show()
