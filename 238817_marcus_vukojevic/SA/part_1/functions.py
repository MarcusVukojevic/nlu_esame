# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
from evals import evaluate
from utils_eval import *

def train_loop(data, optimizer, criterion_ote_labels, criterion_ts_labels, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        ote_labels, ts_labels = model(sample['tokens'], sample['attention'])
        loss_ote = criterion_ote_labels(ote_labels, sample['y_ote_labels'])
        loss_ts = criterion_ts_labels(ts_labels, sample['y_ts_labels'])

        loss = loss_ote + loss_ts
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array



def eval_loop(data, criterion_ote_labels, criterion_ts_labels, model, lang, tokenizer):
    model.eval()
    loss_array = []
    
    ref_slots_ote = []
    hyp_slots_ote = []


    ref_slots_ts = []
    hyp_slots_ts = []

    with torch.no_grad():  # Evita la creazione del grafico computazionale
        for sample in data:
        
            
            ote_labels, ts_labels = model(sample['tokens'], sample['attention'])
            loss_ote = criterion_ote_labels(ote_labels, sample['y_ote_labels'])
            loss_ts = criterion_ts_labels(ts_labels, sample['y_ts_labels'])

            loss = loss_ote + loss_ts
            loss_array.append(loss.item())
                
            # Inferenza delle etichette predette
            ote_output_labels = torch.argmax(ote_labels, dim=1) 
            ts_output_labels = torch.argmax(ts_labels, dim=1)  
            #print(output_labels)

            # PARTE OTE
            for id_seq in range(len(ote_output_labels)):
                length = sample['y_ote_lengths'][id_seq].tolist()  # Lunghezza reale della sequenza
                #utt_ids = sample['tokens'][id_seq][:length].tolist()
                gt_ids = sample['y_ote_labels'][id_seq][:length].tolist()
                gt_slots = [lang.id_2_ote[elem] for elem in gt_ids]
                #utterance = [tokenizer.convert_ids_to_tokens(tok) for tok in utt_ids]
                to_decode = ote_output_labels[id_seq][:length].tolist()
                #print(to_decode, [lang.id2lables[elem] for elem in to_decode])
                ref_slots_ote.append(gt_slots)
                hyp_slots_ote.append([lang.id_2_ote[elem] for elem in to_decode])
            
            # PARTE TS
            for id_seq in range(len(ts_output_labels)):
                length = sample['y_ts_lengths'][id_seq].tolist()  # Lunghezza reale della sequenza
                #utt_ids = sample['tokens'][id_seq][:length].tolist()
                gt_ids = sample['y_ts_labels'][id_seq][:length].tolist()
                gt_slots = [lang.id_2_ts[elem] for elem in gt_ids]
                #utterance = [tokenizer.convert_ids_to_tokens(tok) for tok in utt_ids]
                to_decode = ts_output_labels[id_seq][:length].tolist()
                #print(to_decode, [lang.id2lables[elem] for elem in to_decode])
                ref_slots_ts.append(gt_slots)
                hyp_slots_ts.append([lang.id_2_ts[elem] for elem in to_decode])
            
    try:
       
        ot_2_bieos_hyp_ote = [ot2bieos_ote(i) for i in hyp_slots_ote]
        ot_2_bieos_ref_ote = [ot2bieos_ote(i) for i in ref_slots_ote]


        ot_2_bieos_hyp_ts = [ot2bieos_ts(i) for i in hyp_slots_ts]
        ot_2_bieos_ref_ts = [ot2bieos_ts(i) for i in ref_slots_ts]

        # di ote mi ritorna (precision, recall, f1) e di ts (ts_macro_f1, ts_micro_p, ts_micro_r, ts_micro_f1)
        results = evaluate(ot_2_bieos_ref_ote, ot_2_bieos_ref_ts, ot_2_bieos_hyp_ote, ot_2_bieos_hyp_ts)
        
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots_ote])
        hyp_s = set([x[1] for x in hyp_slots_ote])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    
    return results, loss_array
