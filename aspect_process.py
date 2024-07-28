

import os
from pathlib import Path
import pickle
from collections import defaultdict, Counter
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('--index',type=int)
args=parser.parse_args()

DATA_ROOT = Path("../nete_format_data")
DATASET = "yelp"
INDEX = str(args.index)
DATA_PATH = DATA_ROOT / DATASET /  "reviews.pickle"

index_dir = DATA_ROOT / DATASET / INDEX
output_dir = Path("/home/P76114511/projects/ERRA/data") / DATASET / INDEX / "train-data"


def get_factorized_id(train_reviews: List[Dict]):
    df = pd.DataFrame(train_reviews)
    encoded_user = pd.factorize(df["user"])[0]
    encoded_item = pd.factorize(df["item"])[0]
    print(f"peeking encoded_user: {encoded_user[:5]}")
    print(f"peeking encoded_item: {encoded_item[:5]}")

    user_dict = dict(zip(df["user"], encoded_user))
    item_dict = dict(zip(df["item"], encoded_item))
    print(f"User count: {len(user_dict)}")
    print(f"Item count: {len(item_dict)}")
    # pickle dump
    with open(output_dir / "user_dict.pickle", "wb") as f:
        pickle.dump(user_dict, f)
    with open(output_dir / "item_dict.pickle", "wb") as f:
        pickle.dump(item_dict, f)
    return user_dict, item_dict

def get_train_reviews():

    print('loading reviews...')
    with open(DATA_ROOT / DATASET / "reviews.pickle", "rb") as f:
        reviews = pickle.load(f)
    print('loading index...')
    with open(os.path.join(index_dir, "train.index"), "r") as f:
        index = [int(x) for x in f.readline().split(" ")]
    index = set(index)
    train_reviews = []

    # we need to first get all reviews in training set
    for idx, review in enumerate(reviews):
        if idx in index:
            train_reviews.append(review)
    print(f"# of train_reviews: {len(train_reviews)}")
    # and then we can o the factorization to get each user/item's encoded id (0,1,2,3...)
    user_dict, item_dict = get_factorized_id(reviews)
    return train_reviews, user_dict, item_dict



def get_aspect_top2(data_path):
    assert os.path.exists(data_path)
    train_reviews, user_dict, _ = get_train_reviews()
    nuser = len(user_dict)


    user_aspect_dict = [[] for _ in range(len(user_dict))]
    pbar = tqdm(train_reviews)
    for review in pbar:
        feat,_,_,_,_ = review["template"]
        user_id = review["user"]
        factorized_id = user_dict[user_id]
        pbar.set_description(f"user_id: {factorized_id}, feat: {feat}")
        user_aspect_dict[factorized_id].append(feat)
    # take the top 2 aspect for every user
    user_aspect_top2 = [[] for _ in range(nuser)]

    for fuid in range(nuser):

        top2 = Counter(
            user_aspect_dict[fuid]
        ).most_common(2)
        # top2 = [(aspect, count), (aspect, count)]
        # [('portions', 6), ('table', 3)]
        user_aspect_top2[fuid] = [x[0] for x in top2]
    return user_aspect_top2


def main():
    user_aspect_top2 = get_aspect_top2(DATA_PATH) #dict
    # random printout some users' top 2 aspect

    print('sample:')
    for i in range(5):
        print(user_aspect_top2[i])
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "user_aspect_top2.pickle", "wb") as f:
        pickle.dump(user_aspect_top2, f)


if __name__ == "__main__":
    main()
