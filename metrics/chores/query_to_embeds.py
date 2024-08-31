import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import json

MAX_SAMPLES = 10000
ENCODER_NAME = "all-mpnet-base-v2"


def load_json_or_jsonl(filepath: str):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(filepath, "r") as f:
            data = [json.loads(line) for line in f]
    return data
def clean_func(text: str):
    # before the first <eos>
    text = text.split("<eos>")[0]
    # remove <bos>, <pad>, <feat> ...
    text = text.replace("<bos>", "").replace("<pad>", "").replace("<feat>", "")
    return text


def main():
    input_filepath = "/home/P76114511/projects/aspect_retriever/checkpoints/dbloss_nomerged_noff/yelp/1/generated_supervised_k=3.jsonl"
    print(f"Loading data from {input_filepath}")
    data = load_json_or_jsonl(input_filepath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fake_table = [clean_func(d["fake"]) for d in data]
    golden_table = [d["real"] for d in data]
    sentence_transformer = SentenceTransformer(ENCODER_NAME)
    fake_table = sentence_transformer.encode(
        fake_table, device=device, convert_to_numpy=True, show_progress_bar=True
    )
    golden_table = sentence_transformer.encode(
        golden_table, device=device, convert_to_numpy=True, show_progress_bar=True
    )
    input_parent = Path(input_filepath).parent
    fake_table_path = os.path.join(input_parent, "query.npy")
    golden_table_path = os.path.join(input_parent, "golden.npy")
    print(f"Saving fake_table to {fake_table_path}, shape: {fake_table.shape}")
    np.save(fake_table_path, fake_table)
    print(f"Saving golden_table to {golden_table_path}, shape: {golden_table.shape}")
    np.save(golden_table_path, golden_table)


if __name__ == "__main__":
    main()
