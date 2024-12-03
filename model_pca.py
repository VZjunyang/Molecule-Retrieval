import sys
from datetime import datetime

# Redirect stdout to both console and a file

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create results file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/results_{timestamp}.txt"
sys.stdout = Logger(log_filename)

# Original script starts here
print("Training started...")


from utils.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
# from models.Model import BaseModel
from models.model2 import GAT_MLP
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.variables import ROOT_DIR

torch.cuda.empty_cache()

train = pd.read_csv(ROOT_DIR + '/data/train.tsv', sep='\t', header=None)
sample_train = train.sample(6000, random_state=42)
sample_train.to_csv(ROOT_DIR + '/data/sample_train.tsv', sep='\t', header=False, index=False)

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='sample_train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 100
batch_size = 64
learning_rate = 2e-5

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = GAT_MLP(model_name=model_name, num_node_features=300, nout=256, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
model.to(device)

### getting U, S, V
print("Calculating PCA...")
pca_loader = DataLoader(train_dataset, batch_size=1)
text_embeddings = []
torch.cuda.empty_cache()
for batch in iter(pca_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)    
    text_embeddings.append(model.get_text_encoder()(input_ids, attention_mask).detach().cpu().numpy())
    input_ids.detach()

    attention_mask.detach()

    with torch.cuda.device(device):
        torch.cuda.empty_cache()
embedding_output = torch.tensor(np.array(text_embeddings).reshape(-1, 768))
U, S, V = torch.pca_lowrank(embedding_output, q=256)
U = U.to(device)
S = S.to(device)
V = V.to(device)
print("PCA done!")
###

def projected_embeddings(A):
    return torch.matmul(A, V)

def reconstruct(A):
    return torch.matmul(A, V.T)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

early_stopping_patience = 20
early_stopping_counter = 0

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for batch in train_loader:
        # print(f"Remaining memory: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes")
        torch.cuda.empty_cache()
        # print(f"Remaining memory after emptying: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes")
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        
        x_text = projected_embeddings(x_text)
        current_loss = contrastive_loss(x_graph, x_text)   
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 
    model.eval()       
    val_loss = 0        
    for batch in val_loader:
        # print(f"Remaining memory: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes")
        torch.cuda.empty_cache()
        # print(f"Remaining memory after emptyin: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes")
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        x_text = projected_embeddings(x_text)
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()
    scheduler.step(val_loss)

    if best_validation_loss < val_loss:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping activated!")
            break

    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )

print("Done training!")

# Save the model
torch.save(model.state_dict(), f"models/model_{timestamp}.pt")

# Close the log file
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal