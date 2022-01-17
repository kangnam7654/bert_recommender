from utils.Modules import RecommendModules
import os
import pandas as pd
from pathlib import Path

class Recommend(RecommendModules):
    """
    추천 모듈들을 상속받아 추천을 하는 클래스입니다.
    """
    def __init__(self):
        super().__init__()
        self.root_dir = Path(__file__).parent.parent
        self.id_to_idx, self.idx_to_id = self.id_idx()
        if not os.path.isfile(os.path.join(self.root_dir, 'correlation_table', 'df_corr.csv')):
            self.embedding_map = self.make_embedding()
            self.df = self.embbeding_df_make(self.embedding_map)
            self.df_corr = self.make_corr_df(self.df)
        else:
            self.df_corr = pd.read_csv(os.path.join(self.root_dir, 'correlation_table', 'df_corr.csv'))

    def __len__(self):
        return len(self.df_corr)

    def cal(self, solution_id, top=5):
        idx = self.id_to_idx[solution_id]
        item = self.df_corr.iloc[idx, :]
        item = item.drop([item.index[idx]])
        sort_item = item.sort_values(ascending=False)
        top_table = sort_item[:top]
        top_idx = top_table.index
        top_id = []
        for i in top_idx:
            top_id.append(self.idx_to_id[int(i)])
        top_table.index = top_id
        top_table = pd.DataFrame(top_table)
        top_table.reset_index(inplace=True)
        top_table.columns = ['solution_id', 'similarity']
        return top_table

if __name__ == '__main__':
    pass