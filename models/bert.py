from transformers import BertModel, BertTokenizer


def load_bert(eval=True, device=None):
    model = BertModel.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    
    if device is not None:
        model = model.to(device)
    if eval:
        model.eval()

    return model, tokenizer
