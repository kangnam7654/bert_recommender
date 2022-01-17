import os
import psycopg2
import pandas as pd
from config import config

class BigWaveTables:
    """
    DB에 접속하여 Table을 불러오고 전처리를 하는 클래스입니다.
    """
    def __init__(self):
        self._product, _pc = self.db_connect()

        # category
        self.category, self.solution_category_join, self.solution_category_group = self.category()

        # tag
        self.tag, self.solution_tag_join, self.solution_tag_group = self.tag()

        # product
        self.solution, self.solution_join = self.solution()

        self.dic_solution_join = self.dic = self.solution_join_to_dict()

    @staticmethod
    def db_connect():
        drt = config_db.drt

        #mrs db 접속
        user = config.user
        password = config.password
        host = config.host
        dbname = config.dbname
        port = config.port
        product_connection_string = f"dbname={dbname} user={user} host={host} password={password} port={port}"
        try:
            product = psycopg2.connect(product_connection_string)
            pc = product.cursor()
            return product, pc
        except:
            print("DB 접속 불가")

    def category(self):
        # pandas를 통한 category 조회
        category = pd.read_sql("select * from public.commons_categorymixin ORDER BY id ASC", self._product)
        category = category[['id', 'name', 'category_type']]

        # category column 이름 수정
        category.rename(columns={'id': 'category_id'}, inplace=True)
        category.rename(columns={'name': 'category_name'}, inplace=True)

        # pandas 를 통한 solution category 조회
        solution_category = pd.read_sql("select * from public.solutions_solutioncategory ORDER BY id ASC", self._product)

        # solution category 와 category join (solution category 를 category name 으로 확인할 수 있도록)
        solution_category_join = solution_category.join(category.set_index('category_id')['category_name'],
                                                        on='category_id')
        solution_category_join = solution_category_join[['id', 'category_id', 'solution_id', 'category_name']]

        # solutioncategry 내용을 solution_id로 그룹핑하고 category_name 항목을 합친다
        solution_category_group = solution_category_join.groupby('solution_id', as_index=False).agg({
            'category_name': lambda x: ','.join(x),
        })
        return category, solution_category_join, solution_category_group

    def tag(self):
        # pandas를 통한 tag 조회
        tag = pd.read_sql("select * from public.commons_tagmixin ORDER BY id ASC", self._product)
        tag = tag[['id', 'name', 'meta_data', 'tag_type']]

        # tag 칼럼명 변경
        tag.rename(columns={'id': 'tag_id'}, inplace=True)
        tag.rename(columns={'name': 'tag_name'}, inplace=True)

        # pandas를 통한 solutiontag 조회
        solution_tag = pd.read_sql("select * from public.solutions_solutiontag ORDER BY id ASC", self._product)
        solution_tag = solution_tag.drop(['created_at', 'updated_at'], axis=1)

        # solutiontag와 tag join (solutiontag를 tag name으로 확인할 수 있도록)
        solution_tag_join = solution_tag.join(tag.set_index('tag_id')['tag_name'], on='tag_id')
        solution_tag_join = solution_tag_join.join(tag.set_index('tag_id')['tag_type'], on='tag_id')
        solution_tag_join = solution_tag_join.join(tag.set_index('tag_id')['meta_data'], on='tag_id')

        # tag_name 항목을 string으로 변경. 변경하지 않으면 뒤에서 오류 발생
        solution_tag_join['tag_name'] = solution_tag_join['tag_name'].astype(str)

        # pandas를 통한 solution 조회
        solution = pd.read_sql("select * from public.solutions_solution ORDER BY id ASC", self._product)

        # solution 칼럼명 변경
        solution.rename(columns={'id': 'solution_id'}, inplace=True)
        solution_key = solution[['solution_id', 'title']]

        # solutiontag도 solution_id 기준으로 그룹핑하여 tag_name을 합쳐준다
        solution_tag_group = solution_tag_join.groupby('solution_id', as_index=False).agg({
            'tag_name': lambda x: ','.join(x)
        })
        return tag, solution_tag_join, solution_tag_group

    def solution(self):
        solution = pd.read_sql("select * from public.solutions_solution ORDER BY id ASC", self._product)
        solution.rename(columns={'id': 'solution_id'}, inplace=True)

        # category_group에 tag_group을 solution_id 기준으로 병합
        solution_join = self.solution_category_group.join(self.solution_tag_group.set_index('solution_id')['tag_name'],
                                                          on='solution_id')
        solution_join['keyword'] = solution_join['tag_name'] + "," + solution_join['category_name']

        # (컴마 포함) 특수 문자 및 제거
        for column in solution_join.columns:
            try:
                solution_join[column] = solution_join[column].str.replace(pat=r'[^\w\s]', repl=r',', regex=True)
            except:
                print(f're exception : {column}')

        return solution, solution_join
    
    def solution_join_to_dict(self):
        """
        solution join table을 딕셔너리화 시키는 함수입니다.
        :return:
        """
        dic_solution_join = {}
        for key in self.solution_join['solution_id']:
            dic_solution_join[key] = []

        for idx in self.solution_join.index:
            # solution_join 테이블에서 각 컬럼을 읽음
            key = self.solution_join.loc[idx, 'solution_id']
            categories = self.solution_join.loc[idx, 'category_name']
            tags = self.solution_join.loc[idx, 'tag_name']
            keywords = self.solution_join.loc[idx, 'keyword']

            # 각 컬럼 분리
            categories_split = categories.split(',')
            tags_split = tags.split(',')
            keywords_split = keywords.split(',')

            # 분리한 단어를 딕셔너리에 선형 결합
            for category in categories_split:
                if category not in dic_solution_join[key]:
                    dic_solution_join[key].append(category)

            for tag in tags_split:
                if tag not in dic_solution_join[key]:
                    dic_solution_join[key].append(tag)

            for keyword in keywords_split:
                if keyword not in dic_solution_join[key]:
                    dic_solution_join[key].append(keyword)
        return dic_solution_join

if __name__ == '__main__':
    tables = BigWaveTables().solution_join
