# 通过n-gram实现词频统计
from collections import defaultdict, Counter
import random


def tokenize(text):
    return [char for char in text]


class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)

    def train(self, sentences):
        for sentence in sentences:
            # tokens = list(sentence)
            tokens = tokenize(sentence)
            for i in range(len(tokens) - self.n + 1):
                gram = tuple(tokens[i:i + self.n - 1])
                next_token = tokens[i + self.n - 1]
                self.ngrams[gram][next_token] += 1

    def predict(self, context):
        context = tuple(context)
        if context in self.ngrams:
            return self.ngrams[context].most_common(1)[0][0]
        else:
            return random.choice(list(self.ngrams.keys()))[0]


if __name__ == "__main__":
    # 示例用法
    demo_sentences = [
        "我爱吃苹果",
        "我喜欢吃梨",
        "我喜欢吃螺蛳粉",
        "他不喜欢吃香蕉",
        "她爱吃草莓",
        "我们都喜欢吃西瓜",
        "他们不喜欢吃橙子",
        "我爱吃葡萄",
        "她喜欢吃芒果",
        "他不喜欢吃柚子",
        "我们都爱吃菠萝",
        "他们喜欢吃樱桃",
        "我不喜欢吃柿子",
        "她爱吃荔枝",
        "他喜欢吃龙眼",
        "我们都不喜欢吃榴莲",
        "他们爱吃山竹",
        "我喜欢吃火龙果",
        "她不喜欢吃猕猴桃",
        "他爱吃百香果"
    ]

    ngram_model = NGramModel(2)
    ngram_model.train(demo_sentences)

    # 预测下一个词
    input_context = ["我"]
    for i in range(10):
        next_word = ngram_model.predict(input_context[-1])
        input_context.append(next_word)
        print("".join(input_context))
