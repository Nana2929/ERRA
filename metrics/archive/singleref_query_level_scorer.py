import sys
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Final
import os

import pandas as pd
import numpy as np
from bart_score import BARTScorer
from bert_score import score as bertscore
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from rouge_score import rouge_scorer

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from maple.dataset.dclass import Strategy
from utils import (
    bleu_score,
    load_json,
    load_pickle,
    load_jsonl,
    unique_sentence_percent,
    evaluate_ndcg,
    evaluate_hit_ratio,
)

# https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb
# https://github.com/neulab/BARTScore
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="metrics.log",
)

# constants

QUERY_OUT: Final[str] = "fake"
GT: Final[str] = "real"
PRED_CAT: Final[str] = "fake_category"
GT_CAT: Final[str] = "al_gt_category"  # !!typo!! of "all_gt_category" in retrieve.py

RET_KEY: Final[str] = "item_retrieved"
EPS: Final[float] = 1e-6
TOP_K: Final[int] = 2
DEVICE: Final[str] = "cuda"
BARTCKPT: Final[str] = "facebook/bart-large-cnn"
BERTCKPT: Final[str] = "bert-base-uncased"
BATCH_SIZE: Final[int] = 256


def main(args):
    input_file = Path(args.input_file)
    input_file_name = os.path.basename(input_file).split(".")[0]

    logging.info(f"input file: {input_file}")
    print(f"input file: {input_file}")

    try:
        data = load_json(input_file)
    except:
        data = load_jsonl(input_file)
    tokenizer = Tokenizer(English().vocab)
    # for user-item pair, randomize a token with weight sampling
    data = data[: args.max_samples]

    text_fake = [x[QUERY_OUT] for x in data]
    text_real = [x[GT] for x in data]
    # calc Hit Ratio
    # =============================================
    #  Ranking
    # ==============================================
    if args.model == "maple" and args.strategy != "gt":
        logging.info("Running aspect-based ranking")
        cats_p = [x[PRED_CAT] for x in data]  # list[list[str]]
        cats = [x[GT_CAT] for x in data]  # list[list[str]]
        hit_ratio = evaluate_hit_ratio(cats, cats_p, TOP_K)
        ndcg = evaluate_ndcg(cats, cats_p, TOP_K)
        print("hit_ratio:", hit_ratio)
        print("ndcg:", ndcg)

    tokens_fake = tokenizer.pipe(text_fake)
    tokens_real = tokenizer.pipe(text_real)
    tokens_fake = [[token.text for token in tokens] for tokens in tokens_fake]
    tokens_real = [[token.text for token in tokens] for tokens in tokens_real]

    # =====================================
    retrieved_texts = []
    for i in range(len(data)):
        retrieved_texts.append(" ".join(x["contents"] for x in data[i][RET_KEY]))

    bartscorer = BARTScorer(device=DEVICE, checkpoint=BARTCKPT)

    # * Review-generation Models (Query) *

    bleu_1 = bleu_score(tokens_real, tokens_fake, 1, True)
    bleu_4 = bleu_score(tokens_real, tokens_fake, 4, True)
    rougescorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    rouges = {}
    for i in range(len(tokens_real)):
        rouge = rougescorer.score(target=text_real[i], prediction=text_fake[i])
        for k, v in rouge.items():
            if k not in rouges:
                rouges[k] = []
            rouges[k].append(v.fmeasure)
    rouges = {k: np.mean(v) for k, v in rouges.items()}

    bert_p, bert_r, bert_f = bertscore(
        refs=text_real,
        cands=text_fake,
        model_type=BERTCKPT,
        lang="en",
        verbose=True,
    )
    bert_p = bert_p.mean().item()
    bert_r = bert_r.mean().item()
    bert_f = bert_f.mean().item()
    bart_score = bartscorer.score(srcs=text_real, tgts=text_fake, batch_size=BATCH_SIZE)
    bart_score_ret = bartscorer.score(
        srcs=retrieved_texts, tgts=text_fake, batch_size=BATCH_SIZE
    )

    bart_score = np.mean(bart_score).item()
    bart_score_ret = np.mean(bart_score_ret).item()
    usr, usn = unique_sentence_percent(tokens_fake)

    logging.info(f"=========== {args.model} ===========")
    logging.info(f"strategy: {args.strategy}")
    logging.info(f"BLEU-1: {bleu_1}")
    logging.info(f"BLEU-4: {bleu_4}")
    for k in rouges:
        logging.info(f"\t{k}: {rouges[k]}")
    logging.info(f"BERT-P: {bert_p}")
    logging.info(f"BERT-R: {bert_r}")
    logging.info(f"BERT-F: {bert_f}")
    logging.info(f"BART: {bart_score}")
    logging.info(f"BART-Retrieved: {bart_score_ret}")
    logging.info(f"USR: {usr}, USN: {usn}")
    if args.model == "maple" and args.strategy != "gt":
        logging.info(f"hit ratio: {hit_ratio}")
        logging.info(f"ndcg: {ndcg}")

    output_path = input_file.parent / f"scores_singleref_input={input_file_name}.csv"

    df = pd.DataFrame(
        {
            "BLEU-1": [bleu_1],
            "BLEU-4": [bleu_4],
            "ROUGE-1": [rouges["rouge1"] * 100],
            "ROUGE-2": [rouges["rouge2"] * 100],
            "ROUGE-L": [rouges["rougeL"] * 100],
            "BERT-P": [bert_p],
            "BERT-R": [bert_r],
            "BERT-F": [bert_f],
            "BART": [bart_score],
            "BART-Retrieved": [bart_score_ret],
            "USR": [usr],
            "USN": [usn],
        },
    )
    if args.model == "maple" and args.strategy != "gt":
        df["hit ratio"] = [hit_ratio]
        df["ndcg"] = [ndcg]
    df.to_csv(output_path)


if __name__ == "__main__":
    filename = Path(__file__).name
    parser = ArgumentParser(
        description=f'[{filename}] Calculating scores against all "segments" (as multiple references) \
                            and choose the best matched one.  The input file should be `retrieved_xx.jsonl` or `generated.jsonl` ; the output is a csv file.'
    )
    parser.add_argument(
        "-a", "--auto_arg_by_dataset", help="automatically set args by dataset name"
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=1,
        help="index of the dataset; e.g. 1, 2, 3, 4, 5",
    )
    parser.add_argument("--model", type=str, default="maple")
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="gt",
        help="retrieval strategy; this argument takes effect only when `model` is `maple`",
    )
    parser.add_argument(
        "-m",
        "--max_samples",
        help="max samples to process",
        type=int,
        default=10000,
    )

    MAPLE_CKPT_DIR = (
        "/home/P76114511/projects/aspect_retriever/checkpoints/dset_ver=2_ptr=False"
    )
    OTHER_CKPT_DIR = "/home/P76114511/PRAG/model_baselines/baseline_checkpoints"
    PEPLER_CKPT_DIR = "/home/P76114511/projects/PEPLER/outputs"
    # /home/P76114511/projects/aspect_retriever/recommender/dataset/dclass.py

    args = parser.parse_args()
    assert args.strategy in Strategy
    assert args.model in ["maple", "peter", "nrt", "att2seq", "random", "pepler"]
    print("strategy:", args.strategy)
    if args.model == "maple":
        args.input_file = f"{MAPLE_CKPT_DIR}/supervised/{args.auto_arg_by_dataset}/{args.index}/retrieved_{args.strategy}.jsonl"
    elif args.model == "pepler":
        args.input_file = f"{PEPLER_CKPT_DIR}/{args.index}/{args.auto_arg_by_dataset}/{args.auto_arg_by_dataset}mf/generated.json"
    else:  # other models
        args.input_file = f"{OTHER_CKPT_DIR}/{args.model}/{args.auto_arg_by_dataset}/{args.index}/generated.json"

    args.data_path = f"../nete_format_data/{args.auto_arg_by_dataset}/reviews.pickle"
    args.index_dir = f"../nete_format_data/{args.auto_arg_by_dataset}/{args.index}"
    main(args)
