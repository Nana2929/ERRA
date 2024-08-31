# RUN OUR OWN SYSTEMS
set -e
#   MAPLE_CKPT_DIR = (
#         "/home/P76114511/projects/aspect_retriever/checkpoints/dset_ver=2_ptr=False"
#     )
OTHER_CKPT_DIR="/home/P76114511/PRAG/model_baselines/baseline_checkpoints"
PEPLER_CKPT_DIR="/home/P76114511/projects/PEPLER/outputs"
MAPLE_CKPT_DIR="/home/P76114511/projects/aspect_retriever/checkpoints/retrain"
export CUDA_VISIBLE_DEVICES=3
for dataset in "gest"; do
    for index in 5; do
        # RUN OTHER SYSTEMS
        for model in "att2seq" "random" "nrt" "peter" "pepler"; do
            time=$(date)
            echo "$time Running query-level, $model, $dataset, $index" >> logs/scoring_process
            if [ $model == "pepler" ]; then
                file="${PEPLER_CKPT_DIR}/${index}/${dataset}/${dataset}mf/generated.json"
            elif [ $model == "maple" ]; then
                file="${MAPLE_CKPT_DIR}/${dataset}/${index}/generated_supervised.jsonl"
            else
                file="${OTHER_CKPT_DIR}/${model}/${dataset}/${index}/generated.json"
            fi
            python3 metrics/multiref_query_level_scorer.py --input_file=$file -a=$dataset -i=$index --model=$model\
            --max_samples=10000 --do_diversity --do_factuality --do_feature
        done
    done
done
for dataset in "gest"; do
    for index in 1 2 3; do
        # RUN OTHER SYSTEMS
        for model in "maple"; do
            time=$(date)
            echo "$time Running query-level, $model, $dataset, $index" >> logs/scoring_process
            file="${MAPLE_CKPT_DIR}/${dataset}/${index}/generated_supervised.jsonl"
            python3 metrics/multiref_query_level_scorer.py --input_file=$file -a=$dataset -i=$index --model=$model\
            --max_samples=10000 --do_diversity --do_factuality --do_feature
        done
    done
done
python3 metrics/aggregate_scores.py

