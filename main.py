import argparse
import os

from tqdm import tqdm
import pandas as pd
import torch

from models.bert import load_bert



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=int, default=0)
    args = parser.parse_args()
    return args


def pre_process(data, tokenizer):
    product = data["product"]
    description = data["description"]
    tokenized = tokenizer.encode_plus(
        description,
        max_length=512,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    return product, tokenized


def post_process(out):
    h = out.last_hidden_state
    cls_token = h[:, 0, :]
    cls_token = cls_token.squeeze()
    return cls_token


def main(args):
    df = pd.read_csv(os.path.join("data", "product_description.csv"), index_col=0)
    model, tokenizer = load_bert()
    
    new_dict = {}
    
    best = 0
    best_idx = 0
    
    for idx in tqdm(range(100)):
        product, tokenized = pre_process(df.iloc[idx], tokenizer)
        out = model(tokenized)
        cls_token = post_process(out)
        new_dict[product] = cls_token
    product = new_dict[args.item]
    for key, value in new_dict.items():
        if key == args.item:
            continue
        else:
            sim = torch.cosine_similarity(product, value, dim=0)
            if sim > best:
                best = sim
                best_idx = key
    print(f"We recommend {best_idx}")
        


if __name__ == "__main__":
    args = get_args()
    main(args)
