with open("shakespeare_complete.txt", "r", encoding="utf-8") as file:
    text = file.read()

# print("Length of the dataset in characters:", len(text))
# print("Length of unique charecters:", len(list(set(text))))
# print(text[:100])

#Unique Characters in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(chars)

#Creating a Mapping and a Simple Encoder/Decoder.
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #Input: String, Output: Integers
decode = lambda l: ''.join([itos[i] for i in l]) #Input: Integers, Output: String

# string_ex = "hi there!"
# print(encode(string_ex))
# print(decode(encode(string_ex)))

#Make the data a torch tensor.
import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

#Split into train and validation sets.
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#The max context size.
blocksize = 8

#Target is from 1 to block size + 1 because the target is always trying to predict the next character.
x = train_data[:blocksize]
y = train_data[1:blocksize+1]

#The amount of sequences to train in parallel.
batchsize = 4

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:blocksize+i] for i in ix])
    y = torch.stack([data[i+1:blocksize+i+1] for i in ix])
    return x, y

# batch_ex_x, batch_ex_y = get_batch(split="train")
# print(batch_ex_x)
# print(batch_ex_y)

import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #Each token directly reads off the logits for the next token using the look-up table generated below.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        #All idx and targets are of the shape (B, T).
        logits = self.token_embedding_table(idx) #Logits are of the shape (B, T, C).

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #Get Predictions
            logits, loss = self(idx)
            # print("Before:", logits.shape) #Shape (B, T, C)
            logits = logits[:, -1, :] #Get the last timestep of every batch.
            # print("After:", logits.shape) #Shape (B, C)
            probs = F.softmax(logits, dim=-1) #Convert to probability distribution.
            idx_next = torch.multinomial(probs, num_samples=1) #Get 1 sample so shape idx_next is (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

model = BigramLanguageModel(vocab_size)
# bx, by = get_batch("train")
# logits, loss = model(bx, by)
# print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
# print(logits.shape)
# print(loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batchsize = 32
for steps in range(10000):
    bx, by = get_batch("train")
    logits, loss = model(bx, by)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Loss:", loss.item())
print("Shakespeare text generated:\n")
context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))