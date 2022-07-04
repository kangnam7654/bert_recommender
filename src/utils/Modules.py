import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
from utils.BigWaveTables import BigWaveTables


class RecommendModules(BigWaveTables):
    def __init__(self):
        super().__init__()
        self.raw_data = self.solution_join

    def id_idx(self):
        """
        solution_id 와 index 의 상호 map 생성
        :return:
        """
        id_to_idx = {}
        idx_to_id = {}
        for idx, id in enumerate(self.solution_join['solution_id']):
            id_to_idx[id] = idx
            idx_to_id[idx] = id
        return id_to_idx, idx_to_id

    @staticmethod
    def bert_tokenizer():
        """
        Bert Tokenizer 불러오기
        :return:
        """
        return BertTokenizer.from_pretrained("kykim/bert-kor-base")

    @staticmethod
    def bert_model():
        """
        Bert Model 불러오기
        :return:
        """
        with torch.no_grad():
            model = BertModel.from_pretrained("kykim/bert-kor-base")
        return model

    def make_embedding(self):
        """
        Bert를 통하여 임베딩 진행
        :return:
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_map_b = {}
        model = self.bert_model().to(device)
        tokenizer = self.bert_tokenizer()

        for num, key in enumerate(self.dic.keys()):
            word_pre = self.dic[key]
            word = ','.join(word_pre)
            tokenized = tokenizer.tokenize(word)
            indexed_token = tokenizer.convert_tokens_to_ids(tokenized)
            segments_ids = [1] * len(tokenized)
            tokens_tensor = torch.tensor([indexed_token], dtype=torch.long, device=device)
            segments_tensor = torch.tensor([segments_ids], dtype=torch.long, device=device)
            output = model(tokens_tensor, segments_tensor)
            result = output.pooler_output[0].tolist()
            embedding_map_b[num] = {'solution_id': key,
                                    'vectors': [result]}
            print(f'{num+1}/{len(self.dic.keys())} completed')
        return embedding_map_b

    def embbeding_df_make(self, embedding_map):
        """
        임베딩 데이터 프레임 생성하는 함수
        :param embedding_map:
        :return:
        """
        df = pd.DataFrame(embedding_map).transpose()
        return df

    def make_corr_df(self, df):
        """
        유사도를 계산한 데이터프레임을 생성하는 함수
        :param df:
        :return:
        """
        corr_df = pd.DataFrame(index=df.index, columns=df.index)
        for i in df.index:
            for j in df.index:
                corr_df.iloc[i, j] = float(self.calculate_similarity(df, i, j))
        corr_df.to_csv('df_corr.csv', index=False, encoding='utf-8')
        return corr_df

    @staticmethod
    def visualize(df_corr):
        """
        히트맵 시각화하는 함수
        :param df_corr:
        :return:
        """
        colormap = plt.cm.PuBu
        plt.figure(figsize=(12, 10))
        plt.title('Correlation')
        sns.heatmap(df_corr.astype(float).corr(), linewidths=0, square=False, cmap=colormap, linecolor='white')

    @staticmethod
    def calculate_similarity(df, idx1, idx2):
        """
        유사도를 계산하는 함수
        :param df:
        :param idx1:
        :param idx2:
        :return:
        """
        vector1 = torch.tensor(df.loc[idx1, 'vectors'])
        vector2 = torch.tensor(df.loc[idx2, 'vectors'])

        cos_similarity = torch.cosine_similarity(vector1, vector2)
        cos_similarity = cos_similarity[cos_similarity != 0]
        cos_similarity = torch.mean(cos_similarity)
        return cos_similarity

if __name__ == '__main__':
    pass
