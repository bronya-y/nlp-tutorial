import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# skip_gram
# 通过中间词去预测上下文
# 是说能预测出相同上下文的词就是近似词了吧
class net(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(net, self).__init__()
        self.w = nn.Linear(vocab_size, embedding_size, bias=False)
        self.wt = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input):
        # input [batch_size, vocab_size] one-hot
        embedding = self.w(input)
        # embedding [batch_size, embedding_size]
        output = self.wt(embedding)
        return output

# 转oneHot
def random_bacth():
    random_inputs = []
    random_labels = []
    # 在skip_gram的范围内随机batch_size个索引
    indexs = np.random.choice(range(len(skip_gram)), batch_size, replace=False)
    for index in indexs:
        random_inputs.append(np.eye(vocab_size)[skip_gram[index][0]])
        random_labels.append(skip_gram[index][1])
    return random_inputs,random_labels

if __name__ == '__main__':
    batch_size = 2  # mini-batch size
    embedding_size = 2  # embedding size
    sentences = ["fruit apple fruit", "apple fruit banana", "fruit banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    words = " ".join(sentences).split()
    words = list(set(words))
    word_dict = {w: i for i, w in enumerate(words)}
    vocab_size = len(word_dict)
    skip_gram = []
    word_list = " ".join(sentences).split()
    # 组训练集
    # 前后两个词互为输入和希望的输出
    for i in range(1, len(word_list) - 1):
        target = word_dict[word_list[i]]
        context = [word_dict[word_list[i - 1]], word_dict[word_list[i + 1]]]
        for w in context:
            skip_gram.append([target, w])

    model = net(embedding_size=embedding_size, vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5000):
        input_batch, target_batch = random_bacth()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)
        optimizer.zero_grad()
        output_batch = model(input_batch)
        loss = criterion(output_batch, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(words):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()