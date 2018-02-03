import numpy as np
import copy
import gc
from scipy import spatial
import enchant
import nltk

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
d = enchant.Dict("en_US")

class Net:
    def __init__(self, vocab_size, hidden_size):
        self.bag_length = 0
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = 0.02
        self.loss = 0.0
        self.vocab = {}
        self.layer_1_matrix = self.init_random(vocab_size, hidden_size)
        self.layer_2_matrix = self.init_random(hidden_size, vocab_size)

    def init_random(self, dim_one, dim_two):
        return np.random.rand(dim_one, dim_two)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def activation(self, x):
        return self.sigmoid(x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def loss_func(self, y, out):
        if out[y] == 0: return 1000000
        return -np.log(out[y])

    def fetch_vectors(self):
        return self.layer_1_matrix

    def backprop(self, y, input, output):
        e = copy.deepcopy(output)
        e[y] -= 1.0

        # input-hidden
        t1 = self.learning_rate * np.outer(input, np.matmul(self.layer_2_matrix, e))
        # print "input:",input.shape,"mult:",np.matmul(self.layer_2_matrix, e).shape
        # hidden-output
        t2 = self.learning_rate * np.outer(np.matmul(self.layer_1_matrix.transpose(), input), e.transpose())

        self.layer_1_matrix -= t1
        self.layer_2_matrix -= t2

        return self.loss_func(y, output)

    def train(self, word_ind, y):
        ################### not included bias
        input = np.zeros(vocab_size)
        for i in range(len(word_ind)):
            input[word_ind[i]] += 1.0 / float(len(word_ind))

        output_1 = self.activation(np.matmul(self.layer_1_matrix.transpose(), input))
        output_2 = self.softmax(np.matmul(self.layer_2_matrix.transpose(), output_1))

        self.loss += self.backprop(y, input, output_2)

f = open("full_text_sentences_new.txt")
lines = f.readlines()
words = []
vocab = {}
num_u = 0
v_size = 1000
temp = {}

for line in lines:
    for word in line.strip().split(' '):
        words.append(word)
        if word not in temp: 
            temp[word] = 1
        else: temp[word] += 1

for i in sorted(temp.items(), key=lambda x:x[1], reverse=True):
    if i[0] not in stopwords and d.check(i[0]) == True and i[0].isdigit() == False:
        # print i[0],
        vocab[i[0]] = num_u
        # print i
        num_u += 1
    if num_u >= v_size: break

# print
del temp,lines
gc.collect()

vocab_size = num_u
print "Data reading complete!"
print "len of vocab =", vocab_size

######################## HYPER-PARAMETERS ########################
num_epochs = 5
bag_length = 2
print_loss_after = 20000
hidden_size = 25
max_iters = 100000
##################################################################

model = Net(vocab_size, hidden_size)

model.bag_length = bag_length
model.vocab = vocab
now = 0

for epoch in range(num_epochs):
    itr = 1
    cnt = 1
    while now + bag_length + bag_length + 1 < len(words):
        word_ind = []
        y = 0
        fl = 0
        now += 1

        if now+bag_length >= len(words): print now+bag_length, len(words)
        if words[now+bag_length] not in vocab: continue

        for i in range(now, now+bag_length):
            if words[i] in vocab:
                word_ind.append(vocab[words[i]])

        for i in range(now+bag_length+1, now+bag_length+1+bag_length):
            if words[i] in vocab:
                word_ind.append(vocab[words[i]])
        
        if len(word_ind) == 0: continue
        y = vocab[words[now+bag_length]]
        
        model.train(word_ind, y)
        itr += 1
        cnt += 1

        if itr >= max_iters: break

        if cnt % print_loss_after == 0:
            print "Epoch: "+str(epoch+1) + "/" + str(num_epochs) + "; Iterations: " + str(itr) + "; Current loss:", model.loss
            model.loss = 0
            cnt = 1

word_vec_all = model.fetch_vectors()
vectors = {}
all = []
for i in vocab:
    all.append(i)
    vectors[i] = word_vec_all[vocab[i]]

def cosine(a, b):
    return spatial.distance.cosine(a, b)

# final = {}

# for i in all:
#     for j in all:
#         similarity = cosine(vectors[i], vectors[j])
#         final[i+","+j] = float(similarity)

# cc = 0

# for i in sorted(final.items(), key=lambda x:x[1], reverse=True):
#     cc += 1
#     print i
#     if cc > 1000: break

while 1:
    word = raw_input("Word> ").strip()

    if word not in vectors: 
        print "NHP"
        continue

    m = {}
    for i in all:
        similarity = cosine(vectors[word], vectors[i])
        m[i] = similarity
    print "Top 10 words:"
    ccc = 0
    for i in sorted(m.items(), key=lambda x:x[1], reverse=False):
        print i,
        ccc += 1
        if ccc >= 10: break
    print