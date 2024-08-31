import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import re
from tools.feat_metrics import SubFeatureScorer, FeatureDictionary
from utils import load_index, load_pickle, load_json_or_jsonl

DATA_ROOT = "/home/P76114511/projects/nete_format_data"
FOLD_NUM = 5
FEA_MINLEN = 4
OTHER_CKPT_DIR = "/home/P76114511/PRAG/model_baselines/baseline_checkpoints"
ERRA_CKPT_DIR = "model"
PEPLER_CKPT_DIR = "/home/P76114511/PEPLER/outputs"
MAPLE_CKPT_DIR = (
    "/home/P76114511/projects/aspect_retriever/checkpoints/dbloss_nomerged_noff"
)
MONO_CKPT_DIR = "/home/P76114511/projects/aspect_retriever/baselines/mono"
dummy_words = [
    "and",
    "very",
    "the",
    "is",
    "a",
    "an",
    "it",
    "this",
    "that",
    "of",
    "in",
    "that",
    "are",
    "were",
    "was",
    "food",
]


def clean_func(text: str):
    # before the first <eos>
    text = text.split("<eos>")[0]
    # remove <bos>, <pad>, <feat> ...
    text = text.replace("<bos>", "").replace("<pad>", "").replace("<feat>", "")
    return text


# split from desc_catids into `num_splits` quantiles
def split_catids(freq_file_path, num_splits=4):
    df = pd.read_csv(freq_file_path)
    desc_names = df["category"].values
    desc_catids = df["index"].values
    split_size = len(desc_catids) // num_splits
    splits = []
    split_names = []
    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size
        if i == num_splits - 1:
            splits.append(desc_catids[start:])
            split_names.append(desc_names[start:])
        else:
            splits.append(desc_catids[start:end])
            split_names.append(desc_names[start:end])
    return splits, split_names


def get_output_fname(model: str, input_filename: str):
    if model == "maple":
        pattern = r"generated_(?P<STRATEGY>[^_]+)_k=(?P<TOPK>\d+)\.jsonl"
        match = re.search(pattern, input_filename)
        if match:
            strategy = match.group("STRATEGY")
            topk = match.group("TOPK")
            return f"{model}_{strategy}_k={topk}_subsetFCR.csv"
    else:
        return f"{model}_subsetFCR.csv"


def get_input_fpath(model: str, index: int):
    if model == "maple":
        return (
            Path(MAPLE_CKPT_DIR)
            / args.auto_arg_by_dataset
            / f"{index}"
            / args.input_filename
        )
    elif model == "pepler":
        return (
            Path(PEPLER_CKPT_DIR)
            / f"{index}"
            / args.auto_arg_by_dataset
            / f"{args.auto_arg_by_dataset}mf"
            / args.input_filename
        )
    elif model == "mono":
        return (
            Path(MONO_CKPT_DIR)
            / args.auto_arg_by_dataset
            / f"{index}"
            / "generated.jsonl"
        )
    elif model == "erra":
        return (
            Path(ERRA_CKPT_DIR)
            / args.auto_arg_by_dataset
            / f"{index}"
            / args.input_filename
        )
    else:
        return (
            Path(OTHER_CKPT_DIR)
            / model
            / args.auto_arg_by_dataset
            / f"{index}"
            / args.input_filename
        )


get_longest_feature_triplet = lambda triplets: max(triplets, key=lambda x: len(x[0]))


def main(args):
    data_path = Path(DATA_ROOT) / args.auto_arg_by_dataset / "reviews.pickle"
    reviews = load_pickle(data_path)
    train_item2feature = {
        i: FeatureDictionary(dummy_words, 4) for i in range(1, FOLD_NUM + 1)
    }

    rows = []

    print(f"Processing model {args.model} on dataset {args.auto_arg_by_dataset}")
    feat2cat = {}
    for index in range(1, FOLD_NUM + 1):
        print(f"index: {index}")
        index_dir = Path(DATA_ROOT) / args.auto_arg_by_dataset / f"{index}"
        train_index = load_index(index_dir / "train.index")
        train_index = set(train_index)
        print(f"train set size: {len(train_index)}")

        # ============================
        # iterate through the reviews to collect features and their categories
        # ============================
        for idx, review in enumerate(reviews):
            item_id = review["item"]
            fea, _, _, _, cat = get_longest_feature_triplet(review["triplets"])
            feat2cat[fea] = cat
            if idx not in train_index:
                continue
            train_item2feature[index].add(key=item_id, feature=fea)
        # end of the review loop
    # end of the index(fold) loop

    # ============================
    # Once our train_item2feature is ready, we can start the evaluation
    # ============================
    for index in range(1, FOLD_NUM + 1):
        test_index = load_index(index_dir / "test.index")
        ifpath = get_input_fpath(args.model, index)
        ofname = get_output_fname(args.model, args.input_filename)
        # ============================
        # load from the prediction file
        # ============================
        data = load_json_or_jsonl(ifpath)
        data = data[: args.max_samples]
        # peprocess the data: cleaning up special tokens
        #                     collect the ground truth features
        for idx, d in enumerate(data):
            d["fake"] = clean_func(d["fake"])
            review = reviews[test_index[idx]]
            d["gt_features"] = set([fea for fea, _, _, _, _ in review["triplets"]])
        # ============================
        # instantiate the scorer
        # initialize for each quantile split and all-together
        # ============================
        freq_filepath = (
            Path(DATA_ROOT)
            / args.auto_arg_by_dataset
            / f"{index}"
            / "aspect_category_frequency.csv"
        )
        idss, namess = split_catids(
            freq_file_path=freq_filepath, num_splits=args.q_splits
        )
        fscorer = SubFeatureScorer(
            testdata=data, item2feature=train_item2feature[index], feat2cat=feat2cat
        )
        # calculate for the quantile splits
        row = {}
        for tile in range(args.q_splits):
            fscorer.initialize(legal_cats=namess[tile])
            _fcr = fscorer.calculate()
            print(f"({tile}-th) subset fcr: {_fcr}")
            row[f"{tile}-th quantile fcr"] = _fcr
        all_cats = set([cat for cats in namess for cat in cats])
        fscorer.initialize(legal_cats=all_cats)
        ovall_fcr = fscorer.calculate()
        row["all fcr"] = ovall_fcr
        rows.append(row)
    # end of the index(fold) loop

    # saving the results
    df = pd.DataFrame(rows)
    output_dir = Path(args.output_dir) / args.auto_arg_by_dataset
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / ofname, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_arg_by_dataset", type=str, default="yelp23")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--q_splits", type=int, default=4)
    parser.add_argument("--model", type=str, default="maple")
    parser.add_argument(
        "--input_filename", type=str
    )  # generated_{STRATEGY}_k={TOPK}.jsonl"
    parser.add_argument("--output_dir", type=str, default="./score_outputs/")
    args = parser.parse_args()
    main(args)

#     python3 metrics/run_fcr.py --auto_arg_by_dataset yelp --model erra --input_filename generated.json --output_dir ./score_outputs/
