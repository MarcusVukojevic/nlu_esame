import torch
from conll import evaluate
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterance'], sample['attention'])
        
        loss_intent = criterion_intents(intent, sample['intent'])
        loss_slot = criterion_slots(slots, sample['slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            #print(sample.keys())
            #print(sample["utterance"].shape)
            #exit()
            #print(sample["utterance"].shape)
            #print(sample["attention"].shape)
            slots, intents = model(sample['utterance'], sample['attention'])
            #print(slots.shape)
            #print(intents.shape)
            #print(model(sample['utterance'], sample['attention']))
            loss_intent = criterion_intents(intents, sample['intent'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intent'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                #utterance = [lang.id2word[elem] for elem in utt_ids]
                utterance = [elem for elem in tokenizer.convert_ids_to_tokens(utt_ids)]
                #print("Utterance")
                #print(utterance)
                #print(tokenizer.decode(utt_ids))
                #exit()
                to_decode = seq[:length].tolist()
                #print("partorito dal modello, to dec")
                #print(to_decode)
                #print("ground truth")
                #print(gt_slots)
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                #print("Quello che ho partorito:")
                #print([i[1] for i in tmp_seq])
                #print("\n")
                hyp_slots.append(tmp_seq)
            
                
                
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array