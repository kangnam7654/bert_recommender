from utils.Recommend import Recommend
import pandas as pd
import os


def main(solution_id, top=5, show=True, save=True, savedir='result', filename='result.csv'):
    """

    :param solution_id: 찾을 solution_id를 입력
    :param top: 몇개의 결과를 return 할지 결정하는 변수
    :param show: 출력을 결정하는 변수
    :param save: 결과 저장을 결정하는 변수
    :param savedir: 결과 저장 폴더 지정
    :param filename: 결과 저장 파일 이름
    :return:
    """
    recommend = Recommend()
    top_table = recommend.cal(solution_id=solution_id, top=top)

    if save:
        # 저장경로 설정
        if savedir == 'result':
            savedir = os.path.join(os.path.curdir, 'result')
        else:
            savedir = savedir
        os.makedirs(savedir, exist_ok=True)

        # 파일 이름 확인 후  저장
        if filename[-3:] == 'csv':
            filename = filename
        else:
            filename = f'{filename}.csv'
        top_table.to_csv(os.path.join(savedir, filename), encoding='utf-8')

    if show:  # 결과 print
        print(top_table)

    return top_table


if __name__ == '__main__':
    result = main(solution_id=3, show=True, top=10)