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

select_longest_triplet = lambda triplets: max(triplets, key=lambda x: len(x[0]))
def main(args):
    reviews = load_pickle(args.data_path)
    testindex = load_index(args.index_dir / "test.index")
    generated = load_json_or_jsonl(args.input_file)
    generated = [clean_func(g["fake"]) for g in generated]
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
            text = select_longest_triplet(triplets)[2]
            # text = " ".join([i["text"] for i in triplets])
            gold.append(text)

    out = mauve.compute_mauve(
        p_text=generated,
        q_text=gold,
        device_id=0,
        max_text_length=60,
        verbose=True,
    )
    input_parent = os.path.dirname(args.input_file)
    mauve_file = os.path.join(input_parent, "mauve.log")
    with open(mauve_file, "w") as f:
        f.write(str(out))

if __name__ == "__main__":
    import argparse
    from easydict import EasyDict as edict
    from pathlib import Path
    DATA_PATH = Path("/home/P76114511/projects/nete_format_data")

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

    assert args.auto_arg_by_dataset in ("yelp", "yelp23", "tripadvisor")
    dargs = edict(
        dict(
            data_path= DATA_PATH / f"{args.auto_arg_by_dataset}/reviews.pickle",
            index_dir= DATA_PATH / f"{args.auto_arg_by_dataset}/{index}",
        )
    )

    args = vars(args)
    args.update(dargs)
    args = edict(args)

    main(args)
