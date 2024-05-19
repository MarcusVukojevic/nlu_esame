# Import necessary libraries
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification
from evals import evaluate, evaluate_ote, evaluate_ts
from seqeval.metrics import f1_score, precision_score, recall_score

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side
PAD_TOKEN = 0

# Define valid labels
valid_labels = {'O', 'T-POS', 'T-NEG', 'T-NEU'}

# Provided load_data function
def load_data(path, valid_labels):
    '''
        input: path/to/data.txt
        output: list of dicts with tokens and labels
    '''
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if "####" in line:  # Ensure the line contains the separator
                sentence, tagged_tokens = line.split("####")
                tokens = []
                labels = []
                tagged_tokens = tagged_tokens.strip().split()
                for tagged_token in tagged_tokens:
                    # Use regex to extract token and label
                    match = re.match(r"(.+)=(\S+)", tagged_token)
                    if match:
                        token, label = match.groups()
                        if label in valid_labels:
                            tokens.append(token)
                            labels.append(label)
                        else:
                            print(f"Label '{label}' not recognized in token: {tagged_token}")
                    else:
                        print(f"Skipping malformed token: {tagged_token}")
                # Add the sentence and corresponding labels to the dataset
                dataset.append({'sentence': sentence.strip(), 'tokens': tokens, 'labels': labels})
    return dataset

# Example file paths (adjust these to your actual file paths)
train_file = os.path.join('dataset', 'train.txt')
test_file = os.path.join('dataset', 'test.txt')

# Load raw data
tmp_train_raw = load_data(train_file, valid_labels)
test_raw = load_data(test_file, valid_labels)

# Split train data into train and dev sets
portion = 0.10
train_data, dev_data = train_test_split(tmp_train_raw, test_size=portion, random_state=42, shuffle=True)

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Function to align tokens with their corresponding labels
def allineo_slots(data, tokenizer):
    tokens = data['tokens']
    labels = data['labels']
    tokenized_input = tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True, max_length=128)
    word_ids = tokenized_input.word_ids()
    aligned_labels = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(PAD_TOKEN)
        else:
            aligned_labels.append(labels[word_idx])
    return {'tokens': tokenized_input['input_ids'], 'labels': aligned_labels, 'attention': tokenized_input['attention_mask']}

# Align tokens with labels for train, dev, and test sets
new_train_raw = [allineo_slots(i, tokenizer) for i in train_data]
new_test_raw = [allineo_slots(i, tokenizer) for i in test_raw]
new_dev_data = [allineo_slots(i, tokenizer) for i in dev_data]

# Define Lang class for label encoding
class Lang():
    def __init__(self, intents, PAD_TOKEN, cutoff=0):
        self.PAD_TOKEN = PAD_TOKEN
        self.labels2id = self.lab2id(intents, pad=False)
        self.id2labels = {v: k for k, v in self.labels2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

# Initialize Lang class with labels
lang = Lang(set(sum([line['labels'] for line in new_train_raw + new_test_raw + new_dev_data], [])), PAD_TOKEN)

# Define custom dataset class
class TokensAndLabels(Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.tokens = [x['tokens'] for x in dataset]
        self.labels = [x['labels'] for x in dataset]
        self.attentions = [x['attention'] for x in dataset]
        self.labels_ids = self.mapping_seq(self.labels, lang.labels2id)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {'tokens': torch.tensor(self.tokens[idx]), 'labels': torch.tensor(self.labels_ids[idx]), 'attention': torch.tensor(self.attentions[idx])}
    
    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            res.append([mapper[x] if x in mapper else mapper[self.unk] for x in seq])
        return res

# Create datasets
train_dataset = TokensAndLabels(new_train_raw, lang)
dev_dataset = TokensAndLabels(new_dev_data, lang)
test_dataset = TokensAndLabels(new_test_raw, lang)

# Define collate function
def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention = torch.stack([item['attention'] for item in batch])
    return {'tokens': tokens, 'labels': labels, 'attention': attention}

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

# Initialize the model
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(lang.labels2id))
model.to(device)

# Training parameters
lr = 5e-5
n_epochs = 50
clip = 5  # Gradient clipping
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_labels = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

# Training and evaluation loop
best_f1 = 0
patience = 3

def remove_padding(predictions, true_labels):
    pred_labels = []
    true_labels_list = []
    for pred, true in zip(predictions, true_labels):
        pred_labels.append([p for p, t in zip(pred, true) if t != PAD_TOKEN])
        true_labels_list.append([t for t in true if t != PAD_TOKEN])
    return pred_labels, true_labels_list

def convert_labels_for_eval(tags):
    converted_tags = []
    for seq in tags:
        converted_seq = []
        for tag in seq:
            if tag.startswith('T-'):
                converted_seq.append(tag[2:])
            else:
                converted_seq.append(tag)
        converted_tags.append(converted_seq)
    return converted_tags

for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    
    # Training loop
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['tokens'].long().to(device)
        attention_mask = batch['attention'].long().to(device)
        labels = batch['labels'].long().to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    
    # Evaluation loop
    if (epoch + 1) % 5 == 0:
        model.eval()
        total_eval_loss = 0
        predictions, true_labels = [], []
        
        for batch in dev_loader:
            with torch.no_grad():
                input_ids = batch['tokens'].long().to(device)
                attention_mask = batch['attention'].long().to(device)
                labels = batch['labels'].long().to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_eval_loss += loss.item()
                predictions.extend(torch.argmax(logits, dim=2).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_eval_loss = total_eval_loss / len(dev_loader)
        print(f"Average Validation Loss: {avg_eval_loss:.4f}")
        
        # Calculate F1 Score using seqeval
        pred_labels, true_labels_flat = remove_padding(predictions, true_labels)
        pred_tags = [[lang.id2labels[p] for p in seq] for seq in pred_labels]
        true_tags = [[lang.id2labels[t] for t in seq] for seq in true_labels_flat]
        
        f1 = f1_score(true_tags, pred_tags)
        print(f"F1 Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            patience = 3
        else:
            patience -= 1
        
        if patience <= 0:
            print("Early stopping")
            break

# Final evaluation on the test set
model.eval()
predictions, true_labels = [], []

for batch in test_loader:
    with torch.no_grad():
        input_ids = batch['tokens'].long().to(device)
        attention_mask = batch['attention'].long().to(device)
        labels = batch['labels'].long().to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        predictions.extend(torch.argmax(logits, dim=2).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Remove padding and convert predictions to labels
pred_labels, true_labels_flat = remove_padding(predictions, true_labels)
pred_tags = [[lang.id2labels[p] for p in seq] for seq in pred_labels]
true_tags = [[lang.id2labels[t] for t in seq] for seq in true_labels_flat]

# Convert labels for evaluation
gold_ot = convert_labels_for_eval(true_tags)
pred_ot = convert_labels_for_eval(pred_tags)
gold_ts = convert_labels_for_eval(true_tags)
pred_ts = convert_labels_for_eval(pred_tags)

# Evaluation metrics using provided functions
ote_scores, ts_scores = evaluate(gold_ot, gold_ts, pred_ot, pred_ts)

print(f"OTE Scores: Precision: {ote_scores[0]:.4f}, Recall: {ote_scores[1]:.4f}, F1: {ote_scores[2]:.4f}")
print(f"TS Scores: Macro F1: {ts_scores[0]:.4f}, Micro Precision: {ts_scores[1]:.4f}, Micro Recall: {ts_scores[2]:.4f}, Micro F1: {ts_scores[3]:.4f}")
