from operator import index
import os
from pathlib import Path
import pandas as pd
import os
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

DATA_DIR = os.path.join(Path(__file__).parent, 'data')


def main():
    data = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))
    item_id = data.iloc[:, 1]
    meta = data.iloc[:, 12:]

    item_meta = pd.concat([item_id, meta], axis=1)
    item_meta = item_meta.applymap(lambda x: int(x))
    item_meta.drop_duplicates(inplace=True)
    item_meta.sort_values('item_id', axis=0, inplace=True)
    item_meta.reset_index(drop=True, inplace=True)

    # model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

    Map = {}
    for idx in tqdm(range(len(item_meta))):
        indice = item_meta.loc[idx, :]
        item = indice['item_id']
        meta = indice[indice == 1].index.to_list()

        concat_text = ' and '.join(meta)
        encode_text = tokenizer(concat_text, return_tensors='pt').to('cuda')
        output = model(**encode_text).pooler_output
        Map[item] = {}
        Map[item]['meta'] = meta
        Map[item]['encoding'] = output

    corr_map = pd.DataFrame(index=sorted(list(Map.keys())))
    for col, col_v in tqdm(Map.items()):
        tmp = []
        for row, row_v in Map.items():
            similarity = torch.cosine_similarity(row_v['encoding'], col_v['encoding'])
            tmp.append(similarity.detach().cpu().item())
        corr_map.insert((int(col)-1), col, tmp)
    corr_map.to_csv(os.path.join(DATA_DIR, 'ml_corr_map.csv'), index=False, encoding='utf-8')


# def main(solution_id, top=5, show=True, save=True, savedir='result', filename='result.csv'):
#     """

#     :param solution_id: 찾을 solution_id를 입력
#     :param top: 몇개의 결과를 return 할지 결정하는 변수
#     :param show: 출력을 결정하는 변수
#     :param save: 결과 저장을 결정하는 변수
#     :param savedir: 결과 저장 폴더 지정
#     :param filename: 결과 저장 파일 이름
#     :return:
#     """
#     recommend = Recommend()
#     top_table = recommend.cal(solution_id=solution_id, top=top)

#     if save:
#         # 저장경로 설정
#         if savedir == 'result':
#             savedir = os.path.join(os.path.curdir, 'result')
#         else:
#             savedir = savedir
#         os.makedirs(savedir, exist_ok=True)

#         # 파일 이름 확인 후  저장
#         if filename[-3:] == 'csv':
#             filename = filename
#         else:
#             filename = f'{filename}.csv'
#         top_table.to_csv(os.path.join(savedir, filename), encoding='utf-8')

#     if show:  # 결과 print
#         print(top_table)

#     return top_table


if __name__ == '__main__':
    main()
    