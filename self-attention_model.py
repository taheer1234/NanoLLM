with open("shakespeare_complete.txt", "r", encoding="utf-8") as file:
    text = file.read()

#Unique Characters in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Creating a Mapping and a Simple Encoder/Decoder.
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #Input: String, Output: Integers
decode = lambda l: ''.join([itos[i] for i in l]) #Input: Integers, Output: String

#Make the data a torch tensor.
import torch
data = torch.tensor(encode(text), dtype=torch.long)

#Split into train and validation sets.
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#The max context size.
blocksize = 8
#The amount of sequences to train in parallel.
batchsize = 32

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:blocksize+i] for i in ix])
    y = torch.stack([data[i+1:blocksize+i+1] for i in ix])
    return x, y

import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters for Transformer
n_embed = 32

#Self-Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(blocksize, blocksize))) #Lower triangular matrix for masking to make it a decoder block.
        self.dropout = nn.Dropout(0.2)

    def forward(self, idx):
        B, T, C = idx.shape
        k = self.key(idx) #k is of the shape (B, T, head_size).
        q = self.query(idx) #q is of the shape (B, T, head_size).
        #Compute affinity between the tokens or how much they talk to each other.
        wei = q @ k.transpose(-2,-1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(idx) #v is of the shape (B, T, head_size).
        out = wei @ v #out is of the shape (B, T, head_size).
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(n_heads))
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Each token directly reads off the logits for the next token using the look-up table generated below.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(blocksize, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed)
            )
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        #All idx and targets are of the shape (B, T).
        token_embed = self.token_embedding_table(idx) #Token embedings are of the shape (B, T, n_embed).
        position_embed = self.position_embedding_table(torch.arange(T,device="cpu")) #Position embeddings are of the shape (T, n_embed).        
        true_embed = token_embed + position_embed #Add the token and position embeddings together. The shape becomes (B, T, n_embed).
        true_embed = self.blocks(true_embed)
        logits = self.lm_head(true_embed) #Logits are of the shape (B, T, vocab_size).

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
            idx_cond = idx[:, -blocksize:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] #Get the last timestep of every batch.
            probs = F.softmax(logits, dim=-1) #Convert to probability distribution.
            idx_next = torch.multinomial(probs, num_samples=1) #Get 1 sample so shape idx_next is (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

model = TransformerModel()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
print("Training the model...")
for steps in range(10000):
    bx, by = get_batch("train")
    logits, loss = model(bx, by)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 1000 == 0:
        print(f"Step {steps}: loss = {loss.item():.4f}")

print("Final Loss:", loss.item())
print("Shakespeare text generated:\n")
context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))