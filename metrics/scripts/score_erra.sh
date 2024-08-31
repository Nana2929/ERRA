set -e
ERRA_CKPT_DIR="model"
export CUDA_VISIBLE_DEVICES=0
dataset="yelp23"
model="erra"
# for index in 1 2 3 4 5; do
#     time=$(date)
#     echo "$time Running query-level, $model, $dataset, $index"
#     file="${ERRA_CKPT_DIR}/${dataset}/${index}/generated.json"
#     python3 metrics/run_multiref_query_level.py --input_file=$file -a=$dataset -i=$index --model=$model\
#     --max_samples=10000 --do_diversity --do_feature --output_dir="./score_outputs/"
# done
# default to run 5 folds
# python3 metrics/run_feat_metrics.py --auto_arg_by_dataset=$dataset --model=$model --input_filename="generated.json" --output_dir="./score_outputs/"
# python3 metrics/run_fcr.py --auto_arg_by_dataset=$dataset --model=$model --input_filename="generated.json" --output_dir="./score_outputs/"

# mauve
for dataset in "yelp" "yelp23"; do
    for index in 1 2 3 4 5; do
        time=$(date)
        echo "$time Running mauve, $model, $dataset, $index"
        file="${ERRA_CKPT_DIR}/${dataset}/${index}/generated.json"
        python3 metrics/run_mauve.py --input_file=$file -a=$dataset -i=$index\
        --max_samples=10000
    done
done