import os
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parent
DATA_DIR = os.path.join(ROOT_DIR, "data")


def u_data_process():
    d = {"user_id": [], "item_id": [], "rating": [], "time_stamp": []}
    with open(os.path.join(DATA_DIR, "ml-100k", "u.data"), "r") as f:
        lines = f.readlines()

    for line in lines:
        line_split = line.replace("\n", "").split("\t")
        d["user_id"].append(line_split[0])
        d["item_id"].append(line_split[1])
        d["rating"].append(line_split[2])
        d["time_stamp"].append(line_split[3])

    df = pd.DataFrame(d)
    desc = df.describe()  # check via debug
    null_check = df.isnull().sum()  # check via debug
    df.to_csv(os.path.join(DATA_DIR, "data.csv"), index=False, encoding="utf-8")


def u_user_process():
    d = {"user_id": [], "age": [], "gender": [], "occupation": [], "zip_code": []}

    with open(os.path.join(DATA_DIR, "ml-100k", "u.user"), "r") as f:
        lines = f.readlines()

    for line in lines:
        line_split = line.replace("\n", "").split("|")
        d["user_id"].append(line_split[0])
        d["age"].append(line_split[1])
        d["gender"].append(line_split[2])
        d["occupation"].append(line_split[3])
        d["zip_code"].append(line_split[4])

    df = pd.DataFrame(d)
    desc = df.describe()
    null_check = df.isnull().sum()
    df.to_csv(os.path.join(DATA_DIR, "user.csv"), index=False, encoding="utf-8")


def u_item_process():
    d = {
        "item_id": [],
        "movie_title": [],
        "release_date": [],
        "video_release_date": [],
        "IMDb_url": [],
        "unknown": [],
        "action": [],
        "adventure": [],
        "animation": [],
        "children": [],
        "comedy": [],
        "crime": [],
        "documentary": [],
        "drama": [],
        "fantasy": [],
        "film_noir": [],
        "horror": [],
        "musical": [],
        "mystery": [],
        "romance": [],
        "sci_fi": [],
        "thriller": [],
        "war": [],
        "western": [],
    }

    with open(os.path.join(DATA_DIR, "ml-100k", "u.item"), "br") as f:
        lines = f.readlines()

    for line in lines:
        line_split = str(line).replace('\n', '').split('|')
        if line_split[1] == 'unknown':
            for id, [k, v] in enumerate(d.items()):
                if id == 0:
                    v.append('267')
                else:
                    if k == 'unknown':
                        v.append('1')
                    else:
                        v.append('0')
            continue
        d['item_id'].append(line_split[0].replace('b', '').replace('"', '').replace("'", ''))
        d['movie_title'].append(line_split[1].split(' (')[0])
        d['release_date'].append(line_split[1].split(' (')[1].replace(')', ''))
        d['video_release_date'].append(line_split[2])
        d['IMDb_url'].append(line_split[4])
        d['unknown'].append(line_split[5])
        d['action'].append(line_split[6])
        d['adventure'].append(line_split[7])
        d['animation'].append(line_split[8])
        d['children'].append(line_split[9])
        d['comedy'].append(line_split[10])
        d['crime'].append(line_split[11])
        d['documentary'].append(line_split[12])
        d['drama'].append(line_split[13])
        d['fantasy'].append(line_split[14])
        d['film_noir'].append(line_split[15])
        d['horror'].append(line_split[16])
        d['musical'].append(line_split[17])
        d['mystery'].append(line_split[18])
        d['romance'].append(line_split[19])
        d['sci_fi'].append(line_split[20])
        d['thriller'].append(line_split[21])
        d['war'].append(line_split[22])
        d['western'].append(line_split[23].replace("\\n'", ''))
        
    df = pd.DataFrame(d)
    df = df.applymap(lambda x: x.replace(r'\n"', ''))
    df.to_csv(os.path.join(DATA_DIR, "item.csv"), index=False, encoding="utf-8")

def all_concat():
    data = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'))
    user = pd.read_csv(os.path.join(DATA_DIR, 'user.csv'))
    item = pd.read_csv(os.path.join(DATA_DIR, 'item.csv'))
    data_user = pd.merge(data, user, how='left', on='user_id')
    all = pd.merge(data_user, item, how='left', on='item_id')
    all.to_csv(os.path.join(DATA_DIR, 'all_data.csv'), index=False, encoding='utf-8')


if __name__ == "__main__":
    u_data_process()
    u_user_process()
    u_item_process()
    all_concat()
