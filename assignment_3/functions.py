#  required functions to iumplement the exercise
import torch.nn as nn
import torch
from sklearn.metrics import classification_report
from evals import evaluate, evaluate_ote, evaluate_ts

def train_loop(data, optimizer, criterion_labels, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        labels = model(sample['tokens'], sample['attention'])
        loss = criterion_labels(labels, sample['y_labels'])
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def split_labels(tags):
    ote_labels = []
    ts_labels = []
    for tag in tags:
        if tag.startswith('T-'):
            ote_labels.append('T')
            ts_labels.append(tag[2:])
        else:
            ote_labels.append(tag)
            ts_labels.append(tag)
    return ote_labels, ts_labels


def eval_loop(data, criterion_labels, model, lang, tokenizer):
    model.eval()
    loss_array = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():  # Evita la creazione del grafico computazionale
        for sample in data:
            input_ids = sample['tokens']
            attention_mask = sample['attention']
            labels = model(input_ids, attention_mask)
            
            # Calcola la perdita (opzionale se ti serve solo per la valutazione)
            loss = criterion_labels(labels, sample['y_labels'])
            loss_array.append(loss.item())
            
            # Inferenza delle etichette predette
            output_labels = torch.argmax(labels, dim=1)  # Assumi che labels abbia shape (batch_size, seq_len, num_labels)
            
            for id_seq in range(len(output_labels)):
                length = sample['labels_len'][id_seq].tolist()  # Lunghezza reale della sequenza
                #utt_ids = sample['tokens'][id_seq][:length].tolist()
                gt_ids = sample['y_labels'][id_seq][:length].tolist()
                gt_slots = [lang.id2lables[elem] for elem in gt_ids]
                #utterance = [tokenizer.convert_ids_to_tokens(tok) for tok in utt_ids]
                to_decode = output_labels[id_seq][:length].tolist()
                ref_slots.append(gt_slots)
                hyp_slots.append([lang.id2lables[elem] for elem in to_decode])

    try:
        # Creare un placeholder per TS (sentiment analysis) se non necessario
        gold_ot = [] # --> tutti i T
        gold_ts = [] # tutti i POS, NEG, NEU
        pred_ot = [] # tutti i T
        pred_ts = [] # tutti i POS, NEG, NEU
        for i in hyp_slots:
            new = split_labels(i)
            pred_ot.append(new[0])
            pred_ts.append(new[1])
        for i in ref_slots:
            new = split_labels(i)
            gold_ot.append(new[0])
            gold_ts.append(new[1])
        # Chiamare la funzione evaluate
        results = evaluate(gold_ot, gold_ts, pred_ot, pred_ts)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    
    return results, loss_array