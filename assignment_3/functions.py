#  required functions to iumplement the exercise
import torch.nn as nn
import torch
from sklearn.metrics import classification_report
from evals import evaluate

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
        gold_ts = [['O'] * len(seq) for seq in ref_slots]
        pred_ts = [['O'] * len(seq) for seq in hyp_slots]
        
        # Chiamare la funzione evaluate
        results = evaluate(gold_ts, ref_slots,pred_ts, hyp_slots)

        # Stampare i risultati
        ote_scores, _ = results  # Ignora TS scores se non necessari
        print("Opinion Target Extraction (OTE) metrics:")
        print("Precision:", ote_scores[0])
        print("Recall:", ote_scores[1])
        print("F1-score:", ote_scores[2])
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    
    return results, loss_array