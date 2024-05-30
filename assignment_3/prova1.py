from utils_eval import read_data, set_labels, build_dataset, ot2bieos_ote, ot2bieos_ts
from evals import evaluate_ote, evaluate_ts
import os


prova = set_labels(read_data(os.path.join("dataset", "train.txt")))

caso_studio = prova[0][0]["ote_tags"]
caso_studio1 = prova[0][0]["ts_tags"]
tmp = [['O', 'O', 'B', 'I', 'E', 'O', 'S', 'O', 'B', 'E', 'O']]

tmp2 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'T-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'T-POS', 'T-POS', 'O']

# Trasformazione dei tag TS nel formato BIEOS
transformed_ts = ot2bieos_ts(caso_studio1)

print(caso_studio)
caso_studio = [['O', 'O', 'O', 'B'], ['O', 'O', 'O', 'B']]

tmp = [ot2bieos_ote(i) for i in caso_studio]
# Verifica della trasformazione
#print(transformed_ts)  # Output atteso: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'B-POS', 'E-POS', 'O']

# Verifica della valutazione

#print(evaluate_ts([transformed_ts], [transformed_ts]))

#print(evaluate_ote(tmp, tmp))
#print(ot2bieos_ote(caso_studio))
print(evaluate_ote(tmp, tmp))
print(evaluate_ote(ot2bieos_ote(tmp), ot2bieos_ote(tmp)))
"""
{'sentence': 'I charge it at night and skip taking the cord with me because of the good battery life.', 
'words': ['i', 'charge', 'it', 'at', 'night', 'and', 'skip', 'taking', 'the', 'cord', 'with', 'me', 'because', 'of', 'the', 'good', 'battery', 'life', 'PUNCT'], 
'ote_raw_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'T', 'O', 'O', 'O', 'O', 'O', 'O', 'T', 'T', 'O'], 
'ts_raw_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'T-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'T-POS', 'T-POS', 'O']}

{'sentence': 'I charge it at night and skip taking the cord with me because of the good battery life.',
'words': ['i', 'charge', 'it', 'at', 'night', 'and', 'skip', 'taking', 'the', 'cord', 'with', 'me', 'because', 'of', 'the', 'good', 'battery', 'life', 'PUNCT'], 
'ote_raw_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'T', 'O', 'O', 'O', 'O', 'O', 'O', 'T', 'T', 'O'], 
'ts_raw_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'T-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'T-POS', 'T-POS', 'O'], 
'ote_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O'], 
'ts_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'B-POS', 'I-POS', 'O'], 
'ote_labels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0], 
'ts_labels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 1, 2, 0]
}


{'sentence': 'I charge it at night and skip taking the cord with me because of the good battery life.', 
'words': ['i', 'charge', 'it', 'at', 'night', 'and', 'skip', 'taking', 'the', 'cord', 'with', 'me', 'because', 'of', 'the', 'good', 'battery', 'life', 'PUNCT'],
'ote_raw_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'T', 'O', 'O', 'O', 'O', 'O', 'O', 'T', 'T', 'O'], 
'ts_raw_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'T-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'T-POS', 'T-POS', 'O'], 
'ote_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'S', 'O'], 
'ts_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-NEU', 'O', 'O', 'O', 'O', 'O', 'O', 'B-POS', 'S-POS', 'O'], 
'ote_labels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0],
'ts_labels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 1, 2, 0]}

"""