from transformers import BertModel, BertTokenizer


def load_bert(eval=True, device=None):
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if device is not None:
        model = model.to(device)
    if eval:
        model.eval()

    return model, tokenizer
