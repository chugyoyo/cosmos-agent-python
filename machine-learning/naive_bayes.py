import numpy as np


class CodeSpamDetectorNB:
    """
    基于多项式朴素贝叶斯（Multinomial Naive Bayes）的垃圾代码检测器。
    """

    def __init__(self, alpha=1.0):
        # 拉普拉斯平滑参数 (alpha=1.0 即标准平滑)
        self.alpha = alpha
        self.vocab = []  # 词汇表
        self.class_priors = {}  # 类别先验概率 P(Y)
        self.word_probs = {}  # 词汇条件概率 P(W|Y)

    def _tokenize(self, text):
        """简单的分词器，将代码拆分成单词和特殊符号。"""
        # 匹配字母数字或非空字符
        return [token.lower() for token in text.split() if token]

    def create_vocabulary(self, documents):
        """从所有文档中创建唯一的词汇表。"""
        vocab_set = set()
        for doc in documents:
            vocab_set.update(self._tokenize(doc))
        self.vocab = sorted(list(vocab_set))
        self.vocab_size = len(self.vocab)

    def _get_feature_vector(self, document):
        """将文档转换为词袋模型（Bag-of-Words），统计词频。"""
        vector = np.zeros(self.vocab_size, dtype=int)
        tokens = self._tokenize(document)

        # 查找单词索引并增加计数
        for token in tokens:
            try:
                idx = self.vocab.index(token)
                vector[idx] += 1
            except ValueError:
                # 忽略不在词汇表中的单词
                pass
        return vector

    def fit(self, X_docs, y):
        """训练模型：计算先验概率和条件概率。"""

        # 1. 准备数据和词汇表
        self.create_vocabulary(X_docs)
        X_matrix = np.array([self._get_feature_vector(doc) for doc in X_docs])
        print(X_matrix.shape)

        y = np.array(y)
        self.classes = np.unique(y)

        n_docs = len(X_docs)

        # 2. 训练（计算概率）
        for c in self.classes:
            is_class = (y == c)
            X_c = X_matrix[is_class]

            # --- 2.1 计算先验概率 P(Y=c) ---
            # P(Y=c) = (类别c的文档数) / (总文档数)
            n_c = X_c.shape[0]
            self.class_priors[c] = n_c / n_docs

            # --- 2.2 计算条件概率 P(W|Y=c) ---
            # 统计类别 c 下，每个单词出现的总次数 (包括平滑项)
            # P(w_i | c) = (N_{ci} + alpha) / (N_c + alpha * |V|)

            # 分子：N_{ci} + alpha
            # N_{ci} 是单词 i 在类别 c 所有文档中出现的总次数
            word_counts = np.sum(X_c, axis=0)
            numerator = word_counts + self.alpha

            # 分母：N_c + alpha * |V|
            # N_c 是类别 c 的所有单词的总次数
            total_words_in_class = np.sum(word_counts)
            denominator = total_words_in_class + self.alpha * self.vocab_size

            # 计算并存储对数条件概率 log(P(W|Y=c))
            self.word_probs[c] = np.log(numerator / denominator)

    def predict(self, X_docs):
        """预测新文档的类别。"""
        predictions = []
        for doc in X_docs:
            x = self._get_feature_vector(doc)
            scores = {}

            for c in self.classes:
                # 1. log(P(Y=c))
                log_prior = np.log(self.class_priors[c])

                # 2. log(P(X|Y=c)) = sum(log(P(w_i|Y=c)) * count_i)
                # 使用 np.dot(x, log_probs) 实现对数似然的加权求和
                log_likelihood = np.dot(x, self.word_probs[c])

                # 3. 后验得分 = log(P(Y=c)) + log(P(X|Y=c))
                scores[c] = log_prior + log_likelihood

            # 找到得分最高的类别
            best_class = max(scores, key=scores.get)
            predictions.append(best_class)

        return np.array(predictions)


# --- 演示和测试 ---
# 训练数据：代码片段
X_train = [
    "def calculate_area(r): return 3.14 * r * r",  # 0: 正常代码
    "class MyClass: def __init__(self, x): self.x = x",  # 0: 正常代码
    "__a__ = 1; __a__ = __a__ + 1; exec('import os')",  # 1: 垃圾/可疑代码
    "var1 = 0; var1 = var1 * 10; var1 = var1 / 5",  # 0: 正常代码 (虽然变量名重复，但结构正常)
    "z = 1; while z < 100: z = z * 2; print(z)",  # 0: 正常代码
    "a = 1; b = 2; c = a + b; c = c * 100; c = c - 999; c = c * 0; exec(c)",  # 1: 垃圾/可疑代码
    "exec ('system.os.exit(0)')" # 1：垃圾/可疑代码
]
y_train = [0, 0, 1, 0, 0, 1, 1]

# 测试数据
X_test = [
    "result = function(input) / 2",  # 偏向正常代码
    "x=1; x=x+1; x=x+1; x=x+1; exec('some code')",  # 偏向垃圾代码 (重复和exec)
    "def clean_func(): pass"  # 正常代码
]

# 1. 初始化模型
detector = CodeSpamDetectorNB(alpha=1.0)

# 2. 训练模型
detector.fit(X_train, y_train)
print("模型训练完成，词汇表大小:", detector.vocab_size)
# print("词汇表:", detector.vocab) # 打印词汇表可查看模型学习到的特征

# 3. 进行预测
predictions = detector.predict(X_test)

# 4. 输出结果
print("\n--- 预测结果 ---")
for doc, pred in zip(X_test, predictions):
    status = "【垃圾/可疑代码】" if pred == 1 else "【正常代码】"
    print(f"代码片段: '{doc}'")
    print(f"预测分类: {pred} {status}")
    print("-" * 20)