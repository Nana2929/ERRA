import mauve
import sys
import pandas as pd
from utils import load_index, load_pickle, load_json_or_jsonl
import os


def clean_func(text: str):
    # before the first <eos>
    text = text.split("<eos>")[0]
    # remove <bos>, <pad>, <feat> ...
    text = text.replace("<bos>", "").replace("<pad>", "").replace("<feat>", "")
    return text


def main(args):
    reviews = load_pickle(args.data_path)
    testindex = load_index(args.index_dir)
    generated = load_json_or_jsonl(args.input_file)
    # Clipping
    generated = generated[: args.max_samples]
    testindex = testindex[: args.max_samples]

    gold = []
    if args.auto_arg_by_dataset == "yelp":
        for idx in testindex:
            review = reviews[idx]
            text = review["template"][2]
            gold.append(text)
    elif args.auto_arg_by_dataset == "yelp23":
        for idx in testindex:
            review = reviews[idx]
            triplets = review["triplets"]
            text = " ".join([i["text"] for i in triplets])
            gold.append(text)

    out = mauve.compute_mauve(
        p_text=[clean_func(i) for i in generated],
        q_text=gold,
        device_id=0,
        max_text_length=60,
        verbose=True,
    )
    print(out)
    input_parent = os.path.dirname(args.input_file)
    mauve_file = os.path.join(input_parent, "mauve.log")
    with open(mauve_file, "w") as f:
        f.write(str(out))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Personalized Retriever model inference"
    )
    parser.add_argument(
        "-a", "--auto_arg_by_dataset", type=str, default=None, help="auto argument"
    )
    parser.add_argument(
        "-i", "--index", type=int, default=1, help="index of the dataset"
    )
    parser.add_argument("--input_file", type=str, help="input file path")
    parser.add_argument("--max_samples", type=int, default=10000, help="max samples")
    args = parser.parse_args()
    index = args.index
    # args.data_path = f"../nete_format_data/{args.auto_arg_by_dataset}/reviews.pickle"
    # args.index_dir = f"../nete_format_data/{args.auto_arg_by_dataset}/{args.index}"
    if args.auto_arg_by_dataset is not None:
        from easydict import EasyDict as edict

        assert args.auto_arg_by_dataset in ("yelp", "yelp23", "tripadvisor")
        if args.auto_arg_by_dataset == "yelp":
            dargs = edict(
                dict(
                    data_path="../nete_format_data/yelp/reviews.pickle",
                    index_dir=f"../nete_format_data/yelp/{index}",
                )
            )
        elif args.auto_arg_by_dataset == "yelp23":
            dargs = edict(
                dict(
                    data_path="../nete_format_data/yelp23/reviews.pickle",
                    index_dir=f"../nete_format_data/yelp23/{index}",
                )
            )
    args = vars(args)
    args.update(dargs)

    main(args)
