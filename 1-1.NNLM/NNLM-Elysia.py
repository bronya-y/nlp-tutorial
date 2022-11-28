import torch
import torch.nn as nn
import torch.optim as optim


class net(nn.Module):
    def __init__(self, embeddingSize, vocabSize, hiddenSize):
        super(net, self).__init__()
        self.embeddingSize = embeddingSize
        self.embedding = nn.Embedding(vocabSize, embeddingSize)
        self.lin1 = nn.Linear(embeddingSize * seq_len, hiddenSize)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hiddenSize, vocabSize)
        self.lin3 = nn.Linear(embeddingSize * seq_len, vocabSize)

    # input [ batch_size, seq_len ]
    def forward(self, input):
        # hidden [batch_size, seq_len, embeddingSize] 先转词向量
        hidden = self.embedding(input)
        # 再展平 [batch_size, seq_len * embeddingSize]
        hidden = hidden.view(-1, seq_len * self.embeddingSize)
        # 再进入mlp
        # hidden1 [batch_size, hiddenSize]
        hidden1 = self.lin1(hidden)
        # 过激活函数 [batch_size, hiddenSize]
        hidden2 = self.relu(hidden1)
        # 再过一次mlp [batch_size, nClass]
        hidden3 = self.lin2(hidden2)
        # 输入过一遍Lin [batch_size, nClass]
        hidden4 = self.lin3(hidden)
        # 矩阵相加，对应位置相加
        output = hidden3 + hidden4
        return output


# def make_batch(sentences):


if __name__ == '__main__':
    seq_len = 2
    embeddingSize = 10
    hiddenSize = 16
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    # 获取全部句子里的词
    words = " ".join(sentences).split()
    # 去重
    words = list(set(words))
    # index--word; word--index
    word_dict = {w: i for i, w in enumerate(words)}
    number_dict = {i: w for i, w in enumerate(words)}
    # 类别
    vocabSize = len(word_dict)
    # 网络
    model = net(embeddingSize, vocabSize, hiddenSize)
    # 损失函数 -- [1,0,0,0]--[0.8,0.1,0.1,0.0]---->求损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)
    # batch
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split() # 单句子的词
        input = [word_dict[n] for n in word[:-1]] # 取前n-1个为输入
        target = word_dict[word[-1]] # 取最后一个为目标输出

        input_batch.append(input)
        target_batch.append(target)
        print(input_batch, target_batch)

    # 转tensor
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # 训练 k 轮
    k = 5000
    for epoch in range(k):
        optimizer.zero_grad()
        output_batch = model(input_batch)
        loss = criterion(output_batch,target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim=True)
    predict = predict[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])


