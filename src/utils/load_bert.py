import torch
from transformers import BertModel, BertTokenizer


def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with torch.no_grad():
        model = BertModel.from_pretrained("bert-base-uncased")
    return model, tokenizer