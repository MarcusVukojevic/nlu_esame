#  required functions to iumplement the exercise
import torch.nn as nn
import torch

from evals import evaluate, evaluate_ote, evaluate_ts
from utils_eval import *


def convert_tags_to_bieos(data):
    finale = []
    
    for idx, item in enumerate(data):
        original_ote_tags = item["ote_tags"]
        original_ts_tags = item["ts_tags"]
        
        converted_ote_tags = ot2bieos_ote(original_ote_tags)
        converted_ts_tags = ot2bieos_ts(original_ts_tags)
        
        finale.append({
            "sentence": item["sentence"],
            "words": item["words"],
            "ote_tags": converted_ote_tags,
            "ts_tags": converted_ts_tags,
        })
    
    return finale