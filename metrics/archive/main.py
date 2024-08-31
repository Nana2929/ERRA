
import logging
from pathlib import Path
from easydict import EasyDict as edict
from argparse import ArgumentParser
from utils import (
    bleu_score,
    feature_coverage_ratio,
    feature_detect,
    feature_diversity,
    feature_matching_ratio,
    mean_absolute_error,
    root_mean_square_error,
    rouge_score,
    unique_sentence_percent,
)

# dish match ratio
# sentence in trainset ratio



def make_menu():
    pass


def main():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--auto_arg_by_dataset",
        type=str,
        default="default_value",
        help="automatically decide args by dataset; accepts yelp23, yelp, gest",
    )
    args = parser.parse_args()
    assert args.auto_arg_by_dataset in ["yelp23", "yelp", "gest"]
    ROOT = Path(__file__).parent.parent.parent / "nete_format_data"
    CKPT_DIR = Path("/home/P76114511/projects/aspect_retriever/checkpoints")
    if args.auto_arg_by_dataset == "yelp23":
        dargs = edict(
            data_path=ROOT / "yelp23/reviews.pickle",
            aspect_path=ROOT / "yelp23/aspect_category_index.csv",
            index_dir=ROOT / "yelp23/1",
            corpus_path=ROOT / "retrieval/gest",
            meta_path=ROOT / "yelp23/meta.json",
            item_meta_path=ROOT / "yelp23/yelp_academic_dataset_business.json",
            user_meta_path=ROOT / "yelp23/yelp_academic_dataset_user.json",
            # ==========================
            pepler_checkpoint_dir=CKPT_DIR
            / "dset_ver=2_ptr=False_arreg=1.0/yelp23/run_2",
        )
    elif args.auto_arg_by_dataset == "yelp":
        dargs = edict(
            data_path=ROOT / "yelp/reviews.pickle",
            index_dir=ROOT / "yelp/1",
            aspect_path=ROOT / "yelp/aspect_category_index.csv",
            corpus_path=ROOT / "retrieval/yelp",
            item_meta_path=ROOT / "yelp/item.json",
            user_meta_path=ROOT / "yelp/user.json",
            # ==========================
            pepler_checkpoint_dir=CKPT_DIR
            / "dset_ver=2_ptr=False_arreg=1.0/yelp/run_1",
        )
    else:
        dargs = edict(
            data_path=ROOT / "gest/reviews.pickle",
            index_dir=ROOT / "gest/1",
            aspect_path=ROOT / "gest/aspect_category_index.csv",
            corpus_path=ROOT / "retrieval/gest",
            item_meta_path=None,
            user_meta_path=None,
            # ==========================
            pepler_checkpoint_dir=CKPT_DIR
            / "dset_ver=2_ptr=False_arreg=1.0/gest/run_3",
        )
    args = edict(vars(args))
    args.update(dargs)
    logging.basicConfig(
        level=logging.INFO,
        filename=args.pepler_checkpoint_dir / "retrieval.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(args)
