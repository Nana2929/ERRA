import os
from utils import load_pickle, load_index, load_json_or_jsonl
from pathlib import Path
import json

MAX_SAMPLES = 10000
OTHER_CKPT_DIR = "/home/P76114511/PRAG/model_baselines/baseline_checkpoints"
PEPLER_CKPT_DIR = "/home/P76114511/PEPLER/outputs"
MAPLE_CKPT_DIR = (
    "/home/P76114511/projects/aspect_retriever/checkpoints/dbloss_no_merged"
)
suffix = os.path.basename(MAPLE_CKPT_DIR).split(".")[0]
print(f"suffix: {suffix}")
OUTPUT_DIR = f"./score_outputs/{suffix}"
DATA_ROOT = "/home/P76114511/projects/nete_format_data"


def save_index(index, file):
    # def load_index(index_path):

    # with open(os.path.join(index_path), "r") as f:
    #     index = [int(x) for x in f.readline().split(" ")]
    # return index
    with open(file, "w") as f:
        f.write(" ".join([str(x) for x in index]))


def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)


def main(args):
    reviews = load_pickle(args.data_path)
    train_ids = load_index(Path(args.index_dir) / "train.index")
    test_ids = load_index(Path(args.index_dir) / "test.index")
    try:
        test_data = load_json_or_jsonl(args.test_file)
    except:
        # change json to jsonl or vice versa in filename
        test_data = load_json_or_jsonl(args.test_file + "l")
    warm_users = set()
    warm_items = set()
    for idx in train_ids:
        warm_users.add(reviews[idx]["user"])
        warm_items.add(reviews[idx]["item"])
    print(f"len(warm_users): {len(warm_users)}")
    print(f"len(warm_items): {len(warm_items)}")
    # rewrite a test.index file...?
    new_test_index = []
    new_test_data = []
    for i in range(len(test_data)):
        testd = test_data[i]
        idx_in_reviews = test_ids[i]
        user_id = testd["user_id"]
        item_id = testd["item_id"]
        # if any does not exist in train set, skip
        if user_id not in warm_users or item_id not in warm_items:
            continue
        new_test_index.append(idx_in_reviews)
        new_test_data.append(testd)

    print(f"The original len(test_data): {len(test_data)}")
    print(f"The new len(test_data): {len(new_test_index)}")

    # !! save new test index !!
    test_file_suffix = os.path.basename(args.test_file)
    test_file_suffix = test_file_suffix.split(".")[0]
    test_file_dir = os.path.dirname(args.test_file)
    new_test_file = f"{test_file_suffix}_warm.json"
    save_index(new_test_index, Path(args.index_dir) / "test_warm.index")
    save_json(new_test_data, os.path.join(test_file_dir, new_test_file))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-a", "--auto_arg_by_dataset", type=str)
    parser.add_argument("--model", type=str, default="maple")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("-i", "--index", type=int, default=1)
    parser.add_argument("--test_file", type=str, default="")
    args = parser.parse_args()
    if args.auto_arg_by_dataset:
        args.data_path = f"{DATA_ROOT}/{args.auto_arg_by_dataset}/reviews.pickle"
        args.index_dir = f"{DATA_ROOT}/{args.auto_arg_by_dataset}/{args.index}"
        if args.model == "maple":
            args.test_file = f"{MAPLE_CKPT_DIR}/{args.auto_arg_by_dataset}/{args.index}/generated_supervised_k=3.json"
        elif args.model == "pepler":
            args.test_file = f"{PEPLER_CKPT_DIR}/{args.auto_arg_by_dataset}/{args.index}/generated.json"
        else:
            args.test_file = f"{OTHER_CKPT_DIR}/{args.auto_arg_by_dataset}/{args.index}/generated.json"

    main(args)
