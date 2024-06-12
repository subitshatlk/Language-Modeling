import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scripts import utils
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import Pool
from collections import Counter
import math

from scripts.utils import get_files, convert_files2idx, convert_line2idx 
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(input, k, padding_token=384):
    
    total_subsequences = sum((len(line) - 1) // k + 1 for line in input)
    input_array = np.full((total_subsequences, k), padding_token, dtype=np.int32)
    target_array = np.full((total_subsequences, k), padding_token, dtype=np.int32)

    index = 0
    for data in input:
        for i in range(0, len(data), k):
            end = min(i + k, len(data))
            input = data[i:end]
            target = data[i + 1:end + 1]
            input_array[index, :len(input)] = input
            target_array[index, :len(target)] = target
            if len(target) < k:
                target_array[index, len(target):] = padding_token
            index += 1
    
    input_seq_tensor = torch.tensor(input_array, dtype=torch.long)
    target_seq_tensor = torch.tensor(target_array, dtype=torch.long)
    
    # Create a TensorDataset
    dataset = TensorDataset(input_seq_tensor, target_seq_tensor)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=128) 
    
    return dataloader



class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.fc1 = nn.Linear(hidden_dim, 300)  
        self.reLU = nn.ReLU()
        self.fc2 = nn.Linear(300, output_dim) 

    def forward(self, input, hidden,target=None):

        embedding = self.embedding(input)  
        output, hidden = self.lstm(embedding, hidden)
        prediction = self.reLU(self.fc1(output))
        prediction = self.fc2(prediction)
        return prediction, hidden
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell
    

def weighted_loss(data,vocab):
 
    pad_index = vocab['[PAD]']
    all_indices = [index for sublist in data for index in sublist if index != pad_index]

    index_counts = Counter(all_indices)
    
    count_total = sum(index_counts.values())
    default_weight = 1.0 / len(vocab)
    weighted_tensor = torch.full((len(vocab),), default_weight)
    
    for index in range(len(vocab)):
        if index == pad_index:
            weighted_tensor[index] = 0
        elif index in index_counts:
            count = index_counts[index]
            weight = 1 - (count / count_total)
            weighted_tensor[index] = weight
    
    return weighted_tensor

def train_process(train_loader, dev_loader, criterion, weights, embedding_dim, hidden_dim=200, epochs=5):

    vocab_size = 386
    output_dim = 386
    best_dev_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        print('epoch=',epoch)
        model.train()  
        epoch_loss = 0.0
        batch_count = 0
        

        for inp, lab in train_loader:
            hidden = model.init_hidden(inp.size(0), device)
            optimizer.zero_grad()
            inp, lab = inp.to(device), lab.to(device)

            output, hidden = model(inp, hidden)  
            loss = criterion(output.transpose(1, 2), lab)
            loss.backward()
            optimizer.step()
            hidden = model.detach_hidden(hidden)
            epoch_loss += loss.item()
            batch_count += 1

        average_loss = epoch_loss / batch_count
        print(f'Epoch {epoch}: Average Training Loss: {average_loss}')

        model.eval()  
        with torch.no_grad():
            dev_loss = 0.0
            for inp, lab in dev_loader:
                hidden = model.init_hidden(inp.size(0), device)
                output, hidden = model(inp.to(device), hidden)
                dev_loss += criterion(output.transpose(1, 2), lab.to(device)).item()
            dev_loss /= len(dev_loader)
        
       
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model_path = f'{args.output_dir}/model_2'
            torch.save(model.state_dict(), model_path) 
            print(f'New best model saved at epoch {epoch} with dev loss: {dev_loss}')

    print(f"Training completed with best dev loss: {best_dev_loss}")
    return model

def compute_perplexity_lstm(model, test_loader, weights):
    model.eval()  
    perplexity = 0.0
    total_sequences = 0
    criterion = nn.CrossEntropyLoss(weight=weights, reduction='none', ignore_index=384)

    with torch.no_grad(): 
        for inp, lab in test_loader:
            inp, lab = inp.to(device), lab.to(device)
                       
            hidden = model.init_hidden(inp.size(0), device)                    
            output, hidden = model(inp, hidden)                
            hidden = model.detach_hidden(hidden)    
            loss = criterion(output.transpose(1, 2), lab).to(device)
            
            for i in range(loss.shape[0]):
                sequence_loss = loss[i].mean()  
                sequence_perplexity2 = math.exp(sequence_loss.item())
                perplexity += sequence_perplexity2
                total_sequences += 1

    average_perplexity = perplexity / total_sequences
    return average_perplexity

def load_vocabulary(file_path):
    with open(file_path, 'rb') as f:
        vocabulary = pickle.load(f)
        print("Vocab loaded")
    return vocabulary


def generate_sequence(model, seed_text, char2idx, idx2char, length=200):
    model.eval()  # Switch the model to evaluation mode
    device = next(model.parameters()).device 
    
    hidden, cell = model.init_hidden(1, device)
    
    for char in seed_text:
        char_2idx = torch.tensor([[char2idx[char]]], device=device)          
        _, (hidden, cell) = model(char_2idx, (hidden, cell))
    character = seed_text[-1]
    
    result_text = seed_text  

    for _ in range(length):
        char_2idx = torch.tensor([[char2idx[character]]], device=device)      
        output, (hidden, cell) = model(char_2idx, (hidden, cell))      
        char_scores = output.squeeze().div(0.8).exp()  
        char_probs = torch.multinomial(char_scores, 1).item()       
        character = idx2char[char_probs]
        result_text += character

    return result_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="/scratch/general/vast/u1471339/mp3")
    parser.add_argument('--model_dir', type=str, default="models")
    parser.add_argument('--input_dir', type=str, default="/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp3/mp3_release")
    parser.add_argument('--vocab_file', type=str, default="vocab.pkl")
    parser.add_argument('--learning_rate', type=float, default='0.00001')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--embedding_dim', type=int, default='50')
    parser.add_argument('--epochs', type=int, default='5')
    parser.add_argument('--k', type=int, default="500")
    parser.add_argument('--hidden_dim', type=int, default="200")
    parser.add_argument('--num_layers', type=int, default="1")
    

    args, _ = parser.parse_known_args()

    vocabulary = load_vocabulary("/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp3/mp3_release/data/vocab.pkl")

    train_files = utils.get_files(f'{args.input_dir}/data/train')
    dev_files = utils.get_files(f'{args.input_dir}/data/dev')
    test_files = utils.get_files(f'{args.input_dir}/data/test')
    
    int2char = {i: char for char, i in vocabulary.items()}
    
    padding_token = vocabulary['[PAD]']
    
    train_data = np.array(utils.convert_files2idx(train_files,vocabulary),dtype=np.ndarray)
    dev_data = np.array(utils.convert_files2idx(dev_files,vocabulary),dtype=np.ndarray)
    test_data = np.array(utils.convert_files2idx(test_files,vocabulary),dtype=np.ndarray)
    
    train_data_loader = preprocess_data(train_data,args.k,padding_token)
    dev_data_loader = preprocess_data(dev_data,args.k,padding_token)
    test_data_loader = preprocess_data(test_data,args.k,padding_token)
    print('Data Loaded')
    
    train_weights = weighted_loss(train_data,vocabulary)
    train_weights = train_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=train_weights,ignore_index=vocabulary['[PAD]'])
    model = train_process(train_data_loader, dev_data_loader,criterion, train_weights, args.embedding_dim, 200, 5) 
    
    
    test_perplexity = compute_perplexity_lstm(model, test_data_loader,train_weights)
    print(f'Perplexity: {test_perplexity}')
    
    num_param = sum (p.numel() for p in model.parameters())
    print('num_Param',num_param)

    torch.save(model.state_dict(), "/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp3/mp3_release/model_latest_0.00001_num1.pth")
    
	
    seeds_list = [
        "The little boy was",
        "Once upon a time in",
        "With the target in",
        "Capitals are big cities. For example,",
        "A cheap alternative to"
    ]

    for seed in seeds_list:
        generated = generate_sequence(model, seed,vocabulary,int2char)
        print(f"Seed Sequence: {seed}")
        print(f"Generated Sequence: {generated}\n")
