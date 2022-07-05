import os
from pathlib import Path

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

DATA_DIR = os.path.join(Path(__file__).parent, "data")


def compute_similarity():
    data = pd.read_csv(os.path.join(DATA_DIR, "all_data.csv"))
    item_id = data.iloc[:, 1]
    meta = data.iloc[:, 12:]

    item_meta = pd.concat([item_id, meta], axis=1)
    item_meta = item_meta.applymap(lambda x: int(x))
    item_meta.drop_duplicates(inplace=True)
    item_meta.sort_values("item_id", axis=0, inplace=True)
    item_meta.reset_index(drop=True, inplace=True)

    # model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with torch.no_grad():
        model = BertModel.from_pretrained("bert-base-uncased")

    Map = {}
    for idx in tqdm(range(len(item_meta))):
        indice = item_meta.loc[idx, :]
        item = indice["item_id"]
        meta = indice[indice == 1].index.to_list()

        concat_text = " and ".join(meta)
        encode_text = tokenizer(concat_text, return_tensors="pt")
        output = model(**encode_text).pooler_output
        Map[item] = {}
        Map[item]["meta"] = meta
        Map[item]["encoding"] = output

    corr_map = pd.DataFrame(index=sorted(list(Map.keys())))
    for col, col_v in tqdm(Map.items()): # col starts with 1 ~
        tmp = []
        for _, row_v in Map.items():
            similarity = torch.cosine_similarity(row_v["encoding"], col_v["encoding"])
            tmp.append(similarity.detach().cpu().item())
        corr_map.insert((int(col) - 1), col, tmp)
    corr_map.to_csv(
        os.path.join(DATA_DIR, "ml_corr_map.csv"), index=True, encoding="utf-8"
    )

if __name__ == '__main__':
    compute_similarity()