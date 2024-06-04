
import torch.nn as nn
import torch

class BertFineTune(nn.Module):
    def __init__(self,model, intent_len, slot_len, dropout_prob=0.1, device="cpu"):
        super().__init__()
        self.device = device  # Memorizza il dispositivo come attributo della classe
        self.bert = model.to(device)  # Sposta il modello BERT sul dispositivo specificato
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_len).to(device)  # Sposta il classificatore di intenti sul dispositivo
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, slot_len).to(device)  # Sposta il classificatore di slot sul dispositivo

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slot_logits = self.slot_classifier(sequence_output)
        
        intent_logits = self.intent_classifier(pooled_output)
        #print(intent_logits.shape)
        slot_logits = slot_logits.permute(0, 2, 1)
        return slot_logits, intent_logits
    
