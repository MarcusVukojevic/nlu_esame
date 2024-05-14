from transformers import BertTokenizer

# Carichiamo il tokenizer BERT pre-addestrato
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Definiamo la frase
text = "i 'd like an afternoon flight from atlanta to san francisco with a stopover in denver arriving i 'd say about mealtime"

# Tokenizziamo la frase
tokens = tokenizer.tokenize(text)

#print("Token:", tokens)

# Etichette originali per le parole intere (prima della tokenizzazione)
# Nota: dovresti avere un'etichetta per ogni token esplicito o implicito
label = 'O O O O B-depart_time.period_of_day O O B-fromloc.city_name O B-toloc.city_name I-toloc.city_name O O O O B-stoploc.city_name O O O O B-arrive_time.time_relative B-arrive_time.time'
labels = label.split()
#{'utterance': "", 'slots': , 'intent': 'flight'}


def modifica_slots(tokens, labels):
    # Etichette per i sub-token
    sub_token_labels = []
    word_index = 0  # Indice per tracciare la parola corrente da etichettare

    # Iteriamo su ciascun token generato dal tokenizer
    for token in tokens:
        # Assegniamo l'etichetta alla prima parte del token e le seguenti parti come 'I-' se 'B-' era assegnato
        label_prefix = labels[word_index]
        if token.startswith("##"):  # Il sub-token è una continuazione della parola
            label_prefix = labels[word_index-1]
            label_prefix = label_prefix.replace('B-', 'I-')  # Cambiamo B- in I- per indicare la continuazione
        sub_token_labels.append(label_prefix)
        
        # Non avanziamo all'etichetta successiva finché non incontriamo un token che non è una continuazione
        if token.startswith("##"):
            pass
        else:
            if word_index <= len(tokens) - 1:
                word_index += 1
    return sub_token_labels, len(tokens) == len(sub_token_labels)    

def allineo_slots(text, labels, tokenizer):

    slot_finale = []
    for i, testo in enumerate(text.split()):
        testo_token = tokenizer.tokenize(testo)
        if len(testo_token) != 1:
            for j in range(len(testo_token)):
                if j != 0 and labels[i] != "O":
                    slot_finale.append(labels[i].replace('B-', 'I-'))
                else:
                    slot_finale.append(labels[i])
        else:   
            slot_finale.append(labels[i])
    
    return slot_finale, len(tokenizer.tokenize(text)) == len(slot_finale)


print("Sub-token Labels:", " ".join(allineo_slots(text, labels, tokenizer)[0]))

"""

O O O O O B-depart_time.period_of_day O O B-fromloc.city_name O 
B-toloc.city_name I-toloc.city_name O O O O O B-stoploc.city_name 
O O O O O B-arrive_time.time_relative B-arrive_time.time I-arrive_time.time

"""