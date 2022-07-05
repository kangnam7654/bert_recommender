import os
from pathlib import Path

import pandas as pd

DATA_DIR = os.path.join(Path(__file__).parent, 'data')


def main(item, top_k=10):
    corr_map = pd.read_csv(os.path.join(DATA_DIR, 'ml_corr_map.csv'), index_col=0)
    item_corr = corr_map.loc[item, :]
    item_corr.drop(index=str(item), axis=0, inplace=True)
    item_corr = item_corr.sort_values(ascending=False)
    result = list(map(lambda x: int(x), item_corr[:top_k].index))
    print(result)

if __name__ == '__main__':
    item = 10
    top_k = 10
    main(item=item,
         top_k=top_k)
        