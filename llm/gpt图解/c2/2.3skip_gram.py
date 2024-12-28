# 分词
import jieba

import torch
import torch.nn as nn
import torch.optim as optim


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=2):
        super(SkipGramModel, self).__init__()
        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.linear = nn.Linear(embedding_dim, vocab_size)
        self.input_hidden = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.hidden_output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, center_word):
        hidden = self.input_hidden(center_word)
        out = self.hidden_output(hidden)
        # embeds = self.embeddings(center_word)
        # out = self.linear(embeds)
        return out


def create_skip_gram_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        for idx, word in enumerate(sentence):
            # print("idx:", idx)
            print(max(idx - window_size, 0), min(idx + window_size, len(sentence)))
            for neighbor in sentence[max(idx - window_size, 0): min(idx + window_size, len(sentence))]:
                if neighbor != word:
                    data.append((word, neighbor))
    return data


def one_hot_encoding(word, word_idx):
    tensor = torch.zeros(len(word_idx))
    tensor[word_idx[word]] = 1
    return tensor


if __name__ == "__main__":
    demo_sentences = [
        "小张是老师",
        "小明是一名学生",
        "小冰是学生",
        "小雪是学生",
        "小李是外科医生",
        "小王是软件工程师",
        "小刘是内科医生"
    ]

    # 对句子进行分词
    demo_sentences = [jieba.lcut(sentence) for sentence in demo_sentences]
    print(demo_sentences)

    # 构建词表
    word_list = set()
    word_idx = {}
    idx_word = {}
    for sentence in demo_sentences:
        for word in sentence:
            if word not in word_list:
                word_list.add(word)
    word_list = list(word_list)
    for idx, word in enumerate(word_list):
        word_idx[word] = idx
        idx_word[idx] = word

    print("词表：", word_list)
    print("词到索引的映射：", word_idx)
    print("索引到词的映射：", idx_word)
    print("词表大小：", len(word_list))

    # 构建数据集
    skip_gram_data = create_skip_gram_dataset(demo_sentences)
    print(skip_gram_data)

    # 对数据集进行one-hot编码
    dataset = [(one_hot_encoding(word, word_idx), word_idx[neighbor]) for word, neighbor in skip_gram_data]
    print(dataset)

    # 训练模型
    # Hyperparameters
    embedding_dim = 2
    vocab_size = len(word_list)
    learning_rate = 0.001
    epochs = 3000

    # Initialize model, loss function, and optimizer
    model = SkipGramModel(vocab_size, embedding_dim)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    lose_values = []
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for center, context in skip_gram_data:
            # center_idx = torch.tensor([word_idx[center]], dtype=torch.long)
            X = one_hot_encoding(center, word_idx).float().unsqueeze(0)
            context_idx = torch.tensor([word_idx[context]], dtype=torch.long)

            optimizer.zero_grad()
            output = model(X)
            loss = loss_function(output, context_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(skip_gram_data)}")
            lose_values.append(total_loss / len(skip_gram_data))

    import matplotlib.pyplot as plt

    plt.plot(lose_values)
    plt.show()

    # 输出skip-gram习得的词向量
    for word, idx in word_idx.items():
        feature = model.input_hidden.weight[:idx].detach().numpy()
        print(word, feature)
