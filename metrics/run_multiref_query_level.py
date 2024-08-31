"""
@File    :   query_level_scorer.py
@Time    :   2024/04/18 16:15:14
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   query_level_scorer.py
            - input file: output of `retrieve.py` (`retrieved_{strategy}.jsonl`)
@Ref     :
# https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb
# https://github.com/neulab/BARTScore
"""

from collections import defaultdict
import os
import logging
from typing import Final
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from spacy.lang.en import English
import pandas as pd
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
import sys
from sklearn.metrics import f1_score


sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from tools.bart_score import BARTScorer
from tools.feat_metrics import unique_sentence_ratio_in_user_and_item, FeatureScorer
from utils import (
    load_index,
    load_pickle,
    unique_sentence_percent,
    bleu_score,
    load_json,
    load_jsonl,
    load_test_segments_features,
    evaluate_hit_ratio,
    evaluate_ndcg,
    feature_matching_ratio,
    get_diversity,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# constants

QUERY_OUT: Final[str] = "fake"

PRED_CAT: Final[str] = "fake_category"
GT_CAT: Final[str] = "all_gt_category"  # "all_gt_category" !!typo!! in retrieve.py

RET_KEY: Final[str] = "item_retrieved"
EPS: Final[float] = 1e-6

TOP_K: Final[int] = 20
DEVICE: Final[str] = "cuda"
BARTCKPT: Final[str] = "facebook/bart-large-cnn"
BERTCKPT: Final[str] = "bert-base-uncased"
BATCH_SIZE: Final[int] = 256


def main(args):
    input_file = Path(args.input_file)
    input_file_name = os.path.basename(input_file).split(".")[0]
    output_path = Path(args.output_dir) / args.model / args.auto_arg_by_dataset / str(args.index) / f"{input_file_name}.csv"

    aspect_path = args.aspect_path
    df = pd.read_csv(aspect_path)
    aspect_list = df["category"].to_list()
    aspect2idx = {k: v for v, k in enumerate(aspect_list)}
    naspect = len(aspect_list)

    logging.info(f"input file: {input_file}")
    print(f"input file: {input_file}")
    # check if input file exists
    if not input_file.exists():
        logging.error(f"input file {input_file} does not exist")
        # load with .json suffix
        input_file = input_file.with_suffix(".jsonl")
        logging.info(f"try loading {input_file} instead")
        if not input_file.exists():
            logging.error(f"input file {input_file} does not exist")
            return
    try:
        data = load_json(input_file)
    except:
        data = load_jsonl(input_file)
    reviews = load_pickle(args.data_path)
    item2feature = defaultdict(set)
    train_index = load_index(Path(args.index_dir) / "train.index")
    train_index = set(train_index)
    for idx, review in enumerate(reviews):
        if idx not in train_index:
            continue
        item_id = review["item"]
        for fea, _, _, _, _ in review["triplets"]:
            item2feature[item_id].add(fea)

        # fea, _, _, _, _ = review["template"]
        # item2feature[item_id].add(fea)
    # =============================================
    #  Diversity
    # ==============================================
    user_usr, item_usr = unique_sentence_ratio_in_user_and_item(data)
    tokenizer = Tokenizer(English().vocab)
    # for user-item pair, randomize a token with weight sampling
    data = data[: args.max_samples]

    logging.info("Loading test segments")
    text_refs, feature_refs = load_test_segments_features(args)

    text_fake = [x[QUERY_OUT] for x in data]  # list[str]
    text_refs = text_refs[: args.max_samples]
    text_fake = text_fake[: args.max_samples]

    # =============================================
    #  Ranking
    # ==============================================
    #  and data[0].get(GT_CAT, None)
    if args.model == "maple" and args.strategy != "gt":
        logging.info("Running aspect-based ranking")
        print(data[0])
        cats_p = [x[PRED_CAT] for x in data]  # list[list[str]]
        cats = [x[GT_CAT] for x in data]  # list[list[str]]

        # to multi-hot
        cats_p_multihot = [np.zeros(naspect) for _ in range(len(cats_p))]
        cats_multihot = [np.zeros(naspect) for _ in range(len(cats))]
        for i in range(len(data)):
            for cat in cats_p[i]:
                cat_id = aspect2idx[cat]
                cats_p_multihot[i][cat_id] = 1
            for cat in cats[i]:
                cat_id = aspect2idx[cat]

                cats_multihot[i][cat_id] = 1
        f1 = f1_score(cats_multihot, cats_p_multihot, average="micro")
        hit_ratio = evaluate_hit_ratio(cats, cats_p, TOP_K)
        ndcg = evaluate_ndcg(cats, cats_p, TOP_K)
        print("hit_ratio:", hit_ratio)
        print("ndcg:", ndcg)
        # sys.exit()

    def clean_func(text: str):
        # before the first <eos>
        text = text.split("<eos>")[0]
        # remove <bos>, <pad>, <feat> ...
        text = text.replace("<bos>", "").replace("<pad>", "").replace("<feat>", "")
        return text

    logging.info("Cleaning up the generated text")
    for i in range(len(text_refs)):

        text_fake[i] = text_fake[i]
        # clean up <eos>, <bos>, <pad>
        text_fake[i] = clean_func(text_fake[i])

    assert len(text_refs) == len(text_fake), f"{len(text_refs)} != {len(text_fake)}"

    # ========= tokenizing the predicted query =========
    # list[str]
    tokens_fake = tokenizer.pipe(text_fake)
    tokens_fake = [[token.text for token in tokens] for tokens in tokens_fake]

    # ========= tokenizing the ground-truth segments =========
    # list[list[str]]
    logging.info("Tokenizing ...")

    text_refs_flatten = [x for sublist in text_refs for x in sublist]
    tokens_refs_flatten = tokenizer.pipe(text_refs_flatten)
    tokens_refs_flatten = [
        [token.text for token in tokens] for tokens in tokens_refs_flatten
    ]
    # using tokens_ref_gids to group the same gids together
    curr_idx = 0
    tokens_refs = [[] for _ in range(len(text_refs))]
    for i in range(len(text_refs)):
        for _ in range(len(text_refs[i])):
            tokens_refs[i].append(tokens_refs_flatten[curr_idx])
            curr_idx += 1
    print("First sample")
    print("generated:", text_fake[0])
    print("references (text):", text_refs[0])
    print("references (tokens):", tokens_refs[0])


    # =====================================
    #  Text-quality: BLEU, ROUGE
    # =====================================

    logging.info("Running Text-quality metrics")
    bleu_1 = bleu_score(
        references=tokens_refs, generated=tokens_fake, n_gram=1, smooth=True
    )
    bleu_4 = bleu_score(
        references=tokens_refs, generated=tokens_fake, n_gram=4, smooth=True
    )

    rougescorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    res_rouges = {}
    for i in range(len(text_fake)):
        _rouges = rougescorer.score_multi(targets=text_refs[i], prediction=text_fake[i])
        # {'rouge1': Score(precision=1.0, recall=0.8, fmeasure=0.888888888888889), 'rouge2': Score(precision=1.0, recall=0.75, fmeasure=0.8571428571428571), 'rougeL': Score(precision=1.0, recall=0.8, fmeasure=0.888888888888889)}
        curr_rouges = {k + "-F": v.fmeasure * 100 for k, v in _rouges.items()}
        curr_rouges.update({k + "-P": v.precision * 100 for k, v in _rouges.items()})
        curr_rouges.update({k + "-R": v.recall * 100 for k, v in _rouges.items()})
        for k in curr_rouges:
            if k not in res_rouges:
                res_rouges[k] = []
            res_rouges[k].append(curr_rouges[k])

    for k in res_rouges:
        res_rouges[k] = np.mean(res_rouges[k])

    # =====================================
    #  Factuality: BARTScore
    # =====================================
    if args.do_factuality:
        logging.info("Running factuality metrics")
        bartscorer = BARTScorer(device=DEVICE, checkpoint=BARTCKPT)
        bart_score_sum = 0
        # Assert we have the same number of references
        for i in tqdm(range(len(text_refs))):
            bart_score = bartscorer.multi_ref_score(
                srcs=[text_fake[i]],
                tgts=[text_refs[i]],
                agg="max",
                batch_size=BATCH_SIZE,
            )
        bart_score_sum += bart_score[0]
        bart_score = bart_score_sum / len(text_refs)

    # =====================================
    #  Feature-coverage: BERTScore, FMR
    # =====================================
    if args.do_feature:
        logging.info("Running BERTScore...")
        bert_p, bert_r, bert_f = bertscore(
            refs=text_refs,
            cands=text_fake,
            model_type=BERTCKPT,
            lang="en",
            verbose=True,
        )
        # get dataset-wise average
        bert_p = bert_p.mean().item()
        bert_r = bert_r.mean().item()
        bert_f = bert_f.mean().item()
        fmr = feature_matching_ratio(
            predict_texts=text_fake, test_features=feature_refs
        )
        ifmr, fcr = -1, -1

    # =====================================
    #  Diversity
    # =====================================
    if args.do_diversity:
        usr, usn = unique_sentence_percent(text_fake)
        entr_scores, distinct_scores = get_diversity(text_fake)
        entr_mean = np.mean(list(entr_scores.values()))

    logging.info("=========== Scores ===========")
    logging.info(f"input file: {input_file}")
    logging.info(f"BLEU-1: {bleu_1}")
    logging.info(f"BLEU-4: {bleu_4}")
    # google's rouges are 0 to 1. Make it 0 to 100
    for rouge_type in res_rouges:
        logging.info(f"{rouge_type}: {res_rouges[rouge_type]}")
    if args.do_feature:
        logging.info(f"BERT-P: {bert_p}")
        logging.info(f"BERT-R: {bert_r}")
        logging.info(f"BERT-F: {bert_f}")
        logging.info(f"FMR: {fmr}")
        logging.info(f"item-wise FMR: {ifmr}")
        logging.info(f"feature coverage ratio: {fcr}")
    if args.do_factuality:
        logging.info(f"BART: {bart_score}")
    if args.do_diversity:
        logging.info(f"USR: {usr}, USN: {usn}")
        logging.info(f"user-wise USR: {user_usr}, item-wise USR: {item_usr}")
        logging.info(f"ENTR: {entr_mean}")
        for gram_size in entr_scores:
            logging.info(f"Entr-{gram_size}: {entr_scores[gram_size]}")
        for gram_size in distinct_scores:
            logging.info(f"Distinct-{gram_size}: {distinct_scores[gram_size]}")

    df = pd.DataFrame(
        {
            "BLEU-1": [bleu_1],  # default
            "BLEU-4": [bleu_4],  # default
            "rouge1-F": [res_rouges["rouge1-F"]],  # default
            "rouge2-F": [res_rouges["rouge2-F"]],
            "rougeL-F": [res_rouges["rougeL-F"]],
            "BERT-P": [bert_p if args.do_feature else -1],
            "BERT-R": [bert_r if args.do_feature else -1],
            "BERT-F": [bert_f if args.do_feature else -1],
            "FMR": [fmr if args.do_feature else -1],
            "BART": [bart_score if args.do_factuality else -1],
            "USR": [usr if args.do_diversity else -1],
            "USN": [usn if args.do_diversity else -1],
            "user-wise USR": [user_usr if args.do_diversity else -1],
            "item-wise USR": [item_usr if args.do_diversity else -1],
            "item-wise FMR": [ifmr if args.do_feature else -1],
        },
        index=[args.model],
    )

    if args.do_diversity:
        for gram_size in distinct_scores:
            df[f"Distinct-{gram_size}"] = [distinct_scores[gram_size]]
        df["ENTR"] = [entr_mean]
        for gram_size in entr_scores:
            df[f"Entr-{gram_size}"] = [entr_scores[gram_size]]
    if args.model == "maple" and args.strategy != "gt":
        df["Hit Ratio"] = [hit_ratio]
        df["ndcg"] = [ndcg]
        df["F1"] = [f1]


    # make dir
    if not os.path.exists(output_path.parent):
        os.makedirs(output_path.parent)
    df.to_csv(output_path)
    logging.info(f"output path: {output_path}")


if __name__ == "__main__":
    filename = Path(__file__).name

    parser = ArgumentParser(
        description=f'[{filename}] Calculating scores against all "segments" (as multiple references) \
                            and choose the best matched one.  The input file should be `retrieved_xx.jsonl` or `generated.jsonl` ; the output is a csv file.'
    )

    parser.add_argument("--input_file", type=str, help="input file path")
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
    parser.add_argument("--output_dir", type=str, default="score_outputs")
    # do aspect-ranking on maple anyway; don't do ranking on other models
    # parser.add_argument(
    #     "--do_aspect",
    #     action="store_true",
    #     help="whether to do aspect category-based evaluation",
    # )
    parser.add_argument("--do_factuality", action="store_true")
    parser.add_argument("--do_feature", action="store_true")
    parser.add_argument("--do_diversity", action="store_true")
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
    parser.add_argument(
        "--same_reflen",
        help="whether to cut the query to the same length as the reference",
        action="store_true",
    )

    args = parser.parse_args()
    args.data_path = f"../nete_format_data/{args.auto_arg_by_dataset}/reviews.pickle"
    args.index_dir = f"../nete_format_data/{args.auto_arg_by_dataset}/{args.index}"
    args.aspect_path = (
        f"../nete_format_data/{args.auto_arg_by_dataset}/aspect_category_index.csv"
    )
    main(args)

    # main(args)
