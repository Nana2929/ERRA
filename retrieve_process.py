from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
import os
from typing import List, Dict
import pickle
import pandas as pd
import faiss
import torch

# https://github.com/facebookresearch/faiss/issues/821
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--index", type=int)
parser.add_argument("--dataset", type=str, help='choices: "yelp23", "yelp"')
args = parser.parse_args()

DATA_ROOT = Path("../nete_format_data")
DATASET = args.dataset
FOLD = str(args.index)
index_dir = DATA_ROOT / DATASET / FOLD
print(f"DATASET: {DATASET}, FOLD: {FOLD}")
# make dir
os.makedirs(f"./data/{DATASET}/{FOLD}/train-data", exist_ok=True)


def get_factorized_id(train_reviews: List[Dict]):
    df = pd.DataFrame(train_reviews)
    encoded_user = pd.factorize(df["user"])[0]
    encoded_item = pd.factorize(df["item"])[0]

    user_dict = dict(zip(df["user"], encoded_user))
    item_dict = dict(zip(df["item"], encoded_item))
    print(f"User count: {len(user_dict)}")
    print(f"Item count: {len(item_dict)}")
    return user_dict, item_dict


def get_train_reviews():

    with open(DATA_ROOT / DATASET / "reviews.pickle", "rb") as f:
        reviews = pickle.load(f)
    with open(os.path.join(index_dir, "train.index"), "r") as f:
        index = [int(x) for x in f.readline().split(" ")]
    index = set(index)
    train_reviews = []

    # we need to first get all reviews in training set
    for idx, review in enumerate(reviews):
        if idx in index:
            train_reviews.append(review)
    # and then we can o the factorization to get each user/item's encoded id (0,1,2,3...)
    user_dict, item_dict = get_factorized_id(reviews)
    user_review_ids = [[] for _ in range(len(user_dict))]
    item_review_ids = [[] for _ in range(len(item_dict))]

    for idx, review in enumerate(train_reviews):
        user_id = review["user"]
        item_id = review["item"]
        user = user_dict[user_id]
        item = item_dict[item_id]
        user_review_ids[user].append(idx)
        item_review_ids[item].append(idx)
    return train_reviews, user_dict, item_dict, user_review_ids, item_review_ids


# * sentence transformer *
print("========== Encoding ==========")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2").to("cuda")


# * preparing data *
reviews, user_dict, item_dict, user_review_ids, item_review_ids = get_train_reviews()
review_texts = [review["template"][2] for review in reviews]
re_emb = model.encode(review_texts, show_progress_bar=True)
print(f"save global embbeddings for all train reviews:{re_emb.shape}")
# with open('./data/train-data/each_sent_emb.pkl', "wb") as fOut:
#         pickle.dump({'sentences': reviews, 'embeddings': re_emb}, fOut)
# 全局检索
quantizer = faiss.IndexFlatL2(384)  # the other index
index = faiss.IndexIVFFlat(quantizer, 384, 1000, faiss.METRIC_L2)
index.train(re_emb)
index.add(re_emb)
assert index.is_trained


# * Query Generation *
print("========== Query Generation ==========")
# 对用户所有的评论取平均值，作為該用戶代表 embeddings。商品也一样做法
user_rev_emb = []
item_rev_emb = []
for i in tqdm(range(len(user_review_ids))):
    li_u = []
    for j in range(len(user_review_ids[i])):
        text = review_texts[user_review_ids[i][j]]
        li_u.append(text)
    tttt = model.encode(li_u)
    tttt = torch.tensor(tttt)
    tttt = tttt.sum(dim=0) / len(user_review_ids[i])
    user_rev_emb.append(tttt)
user_rev_emb = torch.tensor([item.detach().numpy() for item in user_rev_emb])
user_rev_emb = user_rev_emb.numpy()
print(user_rev_emb.shape)
torch.save(user_rev_emb, f"./data/{DATASET}/{FOLD}/user_avg_rev_emb.pt")

item_rev_emb = []
for i in tqdm(range(len(item_review_ids))):
    li_u = []
    for j in range(len(item_review_ids[i])):
        text = review_texts[item_review_ids[i][j]]
        li_u.append(text)
    tttt = model.encode(li_u)
    tttt = torch.tensor(tttt)
    tttt = tttt.sum(dim=0) / len(user_review_ids[i])
    item_rev_emb.append(tttt)
item_rev_emb = torch.tensor([item.detach().numpy() for item in item_rev_emb])
item_rev_emb = item_rev_emb.numpy()
print(item_rev_emb.shape)
torch.save(item_rev_emb, f"./data/{DATASET}/{FOLD}/item_avg_rev_emb.pt")


# * Retrieval *
print("========== Retrieval ==========")
print("开始 user")
D_u, I_u = index.search(user_rev_emb, 3)
user_retrive = []
for i in tqdm(range(len(user_review_ids))):
    temp = " ".join([review_texts[I_u[i][j]] for j in range(3)])
    user_temp = model.encode(temp)
    user_retrive.append(user_temp)
user_retrive = torch.tensor(user_retrive)
print("user_retrieve.shape:", user_retrive.shape)

torch.save(user_retrive, f"./data/{DATASET}/{FOLD}/train-data/user_glabos_retrive.pt")


D_i, I_i = index.search(item_rev_emb, 3)
print("开始item")
item_retrive = []
for i in tqdm(range(len(item_review_ids))):
    temp = " ".join([review_texts[I_i[i][j]] for j in range(3)])
    item_temp = model.encode(temp)
    item_retrive.append(item_temp)
item_retrive = torch.tensor(item_retrive)
print("item_retrieve.shape:", item_retrive.shape)
torch.save(item_retrive, f"./data/{DATASET}/{FOLD}/train-data/item_glabos_retrive.pt")
