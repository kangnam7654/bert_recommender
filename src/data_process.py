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
        "movie_id": [],
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
        pass


if __name__ == "__main__":
    u_item_process()
