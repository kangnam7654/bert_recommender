import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import BertModel, BertTokenizer
import pandas as pd


def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with torch.no_grad():
        model = BertModel.from_pretrained("bert-base-uncased")
    return model, tokenizer


def visualize(df_corr):
    """
    히트맵 시각화하는 함수
    :param df_corr:
    :return:
    """
    colormap = plt.cm.PuBu
    plt.figure(figsize=(12, 10))
    plt.title("Correlation")
    sns.heatmap(
        df_corr.astype(float).corr(),
        linewidths=0,
        square=False,
        cmap=colormap,
        linecolor="white",
    )

if __name__ == "__main__":
    model, tokenizer = load_bert()
    text1 = "date and month"
    text2 = "monday and sunday"
    text3 = "mars and pluto"

    encode_input1 = tokenizer(text1, return_tensors="pt")
    encode_input2 = tokenizer(text2, return_tensors="pt")
    encode_input3 = tokenizer(text3, return_tensors="pt")

    output1 = model(**encode_input1)
    output2 = model(**encode_input2)
    output3 = model(**encode_input3)

    cls1 = output1.pooler_output
    cls2 = output2.pooler_output
    cls3 = output3.pooler_output

    a = torch.cosine_similarity(cls1, cls2, dim=-1)
    b = torch.cosine_similarity(cls1, cls3, dim=-1)
    pass
