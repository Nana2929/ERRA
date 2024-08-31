# input paths: /home/P76114511/projects/aspect_retriever/checkpoints/dset_ver=2_ptr=False/supervised/gest/5/scores_input=retrieved_gt.csv
#  print("strategy:", args.strategy)
#     if args.model == "maple":
#         args.input_file = f"{MAPLE_CKPT_DIR}/supervised/{args.auto_arg_by_dataset}/{args.index}/retrieved_{args.strategy}.jsonl"
#     elif args.model == "pepler":
#         args.input_file = f"{PEPLER_CKPT_DIR}/{args.index}/{args.auto_arg_by_dataset}/{args.auto_arg_by_dataset}mf/generated.json"
#     else:  # other models
#         args.input_file = f"{OTHER_CKPT_DIR}/{args.model}/{args.auto_arg_by_dataset}/{args.index}/generated.json"
import pandas as pd
import os

MAPLE_CKPT_DIR = "/home/P76114511/projects/aspect_retriever/checkpoints/retrain"
OTHER_CKPT_DIR = "/home/P76114511/PRAG/model_baselines/baseline_checkpoints"
PEPLER_CKPT_DIR = "/home/P76114511/projects/aspect_retriever/pepler_checkpoints"
# fixed
# File not found: /home/P76114511/projects/PEPLER/outputs/3/yelp/yelpmf/scores_multiref_input=generated.csv

# fixed
# File not found: /home/P76114511/projects/PEPLER/outputs/1/gest/gestmf/scores_multiref_input=generated.csv


# File not found: /home/P76114511/PRAG/model_baselines/baseline_checkpoints/att2seq/yelp/1/scores_multiref_input
MODELS = ["pepler"] # "maple", "random", "att2seq", "nrt", "peter",
DATASETS = ["gest", "yelp", "yelp23"]
STRATEGIES = ["gt", "supervised", "heuristic", "heu_tfidf"]
INDEX_NUM = 5
CSV_PREFIX = "scores_multiref_input="


# ,BLEU-1,BLEU-4,BERT-P,BERT-R,BERT-F,FMR,BART,USR,USN,rouge1-F,rouge2-F,rougeL-F,rouge1-P,rouge2-P,rougeL-P,rouge1-R,rouge2-R,rougeL-R,ENTR,Entr-1,Entr-2,Entr-3,Distinct-1,Distinct-2,Distinct-3
def process_csv(input_csv):
    if os.path.exists(input_csv):
        temp_df = pd.read_csv(input_csv)
        # to dict
        # drop unnamed columns
        temp_df.drop(
            temp_df.columns[temp_df.columns.str.contains("unnamed", case=False)],
            axis=1,
            inplace=True,
        )
        temp_df.insert(0, "dataset", dataset)
        temp_df.insert(1, "model", model)
        temp_df.insert(2, "strategy", strategy)
        temp_df.insert(3, "index", index)
        temp_df = temp_df.T.to_dict()[0]
        return temp_df  # dict

    else:
        print(f"File not found: {input_csv}")


dfs = []
for dataset in DATASETS:
    for model in MODELS:
            if model == "maple":
                for strategy in STRATEGIES:
                    for index in range(1, INDEX_NUM + 1):
                        filename = f"generated_{strategy}"
                        input_csv = (
                            f"{MAPLE_CKPT_DIR}/{dataset}/{index}/{CSV_PREFIX}{filename}.csv"
                        )
                        temp_df = process_csv(input_csv)
                        if temp_df:
                            dfs.append(temp_df)
            else:
                strategy = "x"
                for index in range(1, INDEX_NUM + 1):
                    if model == "pepler":
                        filename = f"generated"
                        input_csv = f"{PEPLER_CKPT_DIR}/{index}/{dataset}/{dataset}mf/{CSV_PREFIX}{filename}.csv"
                    else:
                        # other models
                        filename = "generated"
                        input_csv = f"{OTHER_CKPT_DIR}/{model}/{dataset}/{index}/{CSV_PREFIX}{filename}.csv"
                    # Check if file exists before reading
                    temp_df = process_csv(input_csv)
                    if temp_df:
                        dfs.append(temp_df)


# Dataset - Model(Strategy) - Index --------- (scores) ----------------
# print all dfs


# stacking up
df = pd.DataFrame(dfs)
print(df.head())

# write out
df.to_csv("metrics/multi_query_level_aggregated_scores.csv")
