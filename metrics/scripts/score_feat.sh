dataset="yelp23"
# ======= MAPLE =======
# filename = f"{MAPLE_CKPT_DIR}/{dataset}/{index}/generated_supervised_k={TOPK}.jsonl"
K=3
python3 metrics/run_feat_metrics.py --auto_arg_by_dataset=$dataset \
                --max_samples=10000 \
                --model="maple" \
                --input_filename=generated_supervised_k=$K.jsonl \
               --output_dir="_score_outputs"

# # ======== PEPLER =======
# # saving on A100
# python3 metrics/run_feat_metrics.py --auto_arg_by_dataset=$dataset \
#                 --max_samples=10000 \
#                 --model="pepler" \
#                 --input_filename=generated.jsonl \
#                 --output_dir="_score_outputs"

# # ======== OTHER =======
# # saving on A100
# model="peter"
# python3 metrics/run_feat_metrics.py --auto_arg_by_dataset=$dataset \
#                 --max_samples=10000 \
#                 --model=$model \
#                 --input_filename=generated_wid.jsonl \
#                 --output_dir="_score_outputs"