from datasets import load_dataset
import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import numpy as np


IDtoLabel_UK = {0: 'O', 1: 'B-Person', 2: 'I-Person', 3: 'B-Organization', 4: 'I-Organization', 5: 'B-Location', 6: 'I-Location', 7: 'B-Misc',  8: 'I-Misc'}
IDtoLabel_US = {0:'O', 1:'B-Misc', 2:'B-Date', 3:'I-Date', 4:'B-Person', 5:'I-Person', 6:'B-NORP', 7:'B-Misc', 8:'I-Misc', 9:'B-Misc', 10:'I-Misc', 11:'B-Organization', 12:'I-Organization', 13:'B-Misc', 14:'I-Misc', 15:'B-Misc', 16:'B-Money', 17:'I-Money', 18:'B-Misc', 19:'I-Misc', 20:'B-Facility', 21:'B-Date', 22:'I-Misc', 23:'B-Location', 24:'B-Misc', 25:'I-Misc', 26:'I-NORP', 27:'I-Location', 28:'B-Product', 29:'I-Date', 30:'B-Misc', 31:'I-Misc', 32:'I-Facility', 33:'B-Misc', 34:'I-Product', 35:'I-Misc', 36:'I-Misc'}
LabelToID = {'O': 0, 'B-Date': 1, 'I-Date': 2, 'B-Person': 3, 'I-Person': 4, 'B-Location': 5, 'I-Location': 6, 'B-Facility': 7, 'I-Facility': 8, 'B-Organization': 9, 'I-Organization': 10, 'B-Misc': 11, 'I-Misc': 12, 'B-Money': 13, 'I-Money': 14, 'B-NORP': 15, 'I-NORP': 16, 'B-Product': 17, 'I-Product': 18}

data=load_dataset("BramVanroy/conll2003")

trainDataUK=data['train'].to_pandas().drop(columns=['pos_tags', 'chunk_tags', 'id'])
trainDataUK['ner_tags'] = trainDataUK['ner_tags'].apply(lambda x: [IDtoLabel_UK[i] for i in x])
trainDataUK['ner_tags'] = trainDataUK['ner_tags'].apply(lambda x: [LabelToID[i] for i in x])

valDataUK=data['validation'].to_pandas().drop(columns=['pos_tags', 'chunk_tags', 'id'])
valDataUK['ner_tags'] = valDataUK['ner_tags'].apply(lambda x: [IDtoLabel_UK[i] for i in x])
valDataUK['ner_tags'] = valDataUK['ner_tags'].apply(lambda x: [LabelToID[i] for i in x])

testDataUK=data['test'].to_pandas().drop(columns=['pos_tags', 'chunk_tags', 'id'])
testDataUK['ner_tags'] = testDataUK['ner_tags'].apply(lambda x: [IDtoLabel_UK[i] for i in x])
testDataUK['ner_tags'] = testDataUK['ner_tags'].apply(lambda x: [LabelToID[i] for i in x])

id2label = {v: k for k, v in LabelToID.items()}

def load_tsvs_from_folder(folder_path):
    all_sentences = []
    all_ner_tags = []
    all_regions = []

    search_pattern = os.path.join(folder_path, "*.tsv")
    file_list = glob.glob(search_pattern)
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        region_name = filename.split('_')[0]
        current_sentence = []
        current_tags = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(2):       # Skip the first two lines
                try:
                    next(f)
                except StopIteration:
                    break

            for line in f:
                line = line.strip()

                if not line:    # EMPTY LINE: Sentence is over
                    if current_sentence:
                        all_sentences.append(list(current_sentence))
                        all_ner_tags.append(list(current_tags))
                        all_regions.append(region_name)
                        current_sentence.clear()
                        current_tags.clear()
                    continue 
                    
                parts = line.split('\t')    # TEXT: Extract token and primary tag
                if len(parts) >= 2:
                    token = parts[0].strip()
                    raw_tag = parts[1].strip() # Grab the FIRST tag to avoid nested tags

                    current_sentence.append(token)
                    current_tags.append(raw_tag)

            if current_sentence:        # END OF FILE: Catch the final sentence
                all_sentences.append(list(current_sentence))
                all_ner_tags.append(list(current_tags))
                all_regions.append(region_name)

    df = pd.DataFrame({'tokens': all_sentences, 'ner_tags': all_ner_tags, 'region': all_regions,})
    return df

folder_path = r".\processed_annotated"
dataGlobal = load_tsvs_from_folder(folder_path)

dataGlobal['ner_tags'] = dataGlobal['ner_tags'].apply(lambda x: [LabelToID[i] for i in x])

def build_vocab(datasets_list): # Vocabulary mapping words to integer IDs
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for df in datasets_list:
        for tokens in df['tokens']:
            for word in tokens:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
    return word2idx

class NERDataset(Dataset):      # PyTorch Dataset Class
    def __init__(self, df, word2idx):
        self.sentences = df['tokens'].tolist()
        self.labels = df['ner_tags'].tolist()
        self.word2idx = word2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):     # Convert words to indices, fallback to <UNK> if word is unseen
        word_indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in self.sentences[idx]]
        label_indices = self.labels[idx]
        return torch.tensor(word_indices), torch.tensor(label_indices)

def collate_fn(batch):      # function to pad sentences to the same length in a batch
    sentences, labels = zip(*batch)
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)   # Pad sentences with 0 (<PAD>)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)      # Pad labels with -100 (PyTorch CrossEntropyLoss ignores -100 by default)
    return padded_sentences, padded_labels


class RNN_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RNN_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)     # Word embedding layer
        
        self.lstm = nn.LSTM(            # Bidirectional LSTM
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True)
        
        # Linear layer mapping hidden states to tag classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)        # hidden_dim * 2 because it's bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits
    
def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sentences, labels in dataloader:
            sentences, labels = sentences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(sentences)
            
            # Reshape logits and labels for CrossEntropyLoss
            logits = logits.view(-1, logits.shape[-1])      # logits: (batch_size * seq_len, num_classes)
            labels = labels.view(-1)        # labels: (batch_size * seq_len)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sentences, labels in dataloader:
            sentences, labels = sentences.to(device), labels.to(device)
            logits = model(sentences)
            
            preds = torch.argmax(logits, dim=-1)        # Get the predicted class indices
            
            preds = preds.view(-1).cpu().numpy()        # Flatten to compare
            labels = labels.view(-1).cpu().numpy()
            
            mask = labels != -100           # Filter out the padded tokens (-100)
            all_preds.extend(preds[mask])
            all_labels.extend(labels[mask])
            
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1


def write_predictions_to_iob2(model, df, word2idx, id2label, output_filename, device):
    model.eval()
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for tokens in df['tokens']:
                word_indices = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]   # Convert words to indices and add a batch dimension of 1
                input_tensor = torch.tensor([word_indices]).to(device)
                
                logits = model(input_tensor)
                
                preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()   # .squeeze(0) removes the batch dimension
                
                for token, pred_id in zip(tokens, preds):
                    pred_label = id2label[pred_id]
                    f.write(f"{token}\t{pred_label}\n")
                
                f.write("\n")


train_ww, test_ww = train_test_split(dataGlobal, test_size=0.2, random_state=42)

train_combined = pd.concat([trainDataUK, train_ww], ignore_index=True)

word2idx = build_vocab([trainDataUK, train_ww])     # Build a master vocabulary from combined
vocab_size = len(word2idx)
num_classes = len(LabelToID)

experiments = [
    {"name": "Train: Worldwide | Test: Worldwide", "train": train_ww, "test": test_ww},
    {"name": "Train: English   | Test: Worldwide", "train": trainDataUK, "test": test_ww},
    {"name": "Train: Worldwide | Test: English",   "train": train_ww,  "test": testDataUK},
    {"name": "Train: English   | Test: English",   "train": trainDataUK, "test": testDataUK},
    {"name": "Train: Combined  | Test: Worldwide", "train": train_combined, "test": test_ww},
    {"name": "Train: Combined  | Test: English",   "train": train_combined, "test": testDataUK},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}

for exp in experiments:
    print(f"\n{'='*50}\nRunning: {exp['name']}\n{'='*50}")
    
    train_ds = NERDataset(exp["train"], word2idx)       # Create DataLoaders
    test_ds = NERDataset(exp["test"], word2idx)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    model = RNN_NER(vocab_size, embedding_dim=128, hidden_dim=256, num_classes=num_classes).to(device)  # New model for each experiment
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignores padding
    
    train_model(model, train_dl, optimizer, criterion, device, epochs=5)
    
    f1_score_val = evaluate_model(model, test_dl, device)
    print(f"Result -> F1 Score: {f1_score_val:.4f}")
    
    results[exp['name']] = f1_score_val

    safe_name = exp['name'].replace(":", "").replace(" | ", "_").replace(" ", "_").lower()  # Create file name based on the experiment name
    filename = f"predictions_{safe_name}.iob2"
    
    write_predictions_to_iob2(
        model=model, 
        df=exp['test'], 
        word2idx=word2idx, 
        id2label=id2label, 
        output_filename=filename, 
        device=device
    )

print("\n--- Final Experiment Results (F1 Score) ---")
for name, score in results.items():
    print(f"{name}: {score:.4f}")