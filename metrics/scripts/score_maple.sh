
# DATASET=yelp
set -e
export CUDA_VISIBLE_DEVICES=0
CKPTDIR="checkpoints/dbloss_nomerged_noff"
for fold in 1 2 3 4 5; do
    for DATASET in "yelp"; do
        for STRATEGY in "supervised"; do
            for max_test_aspect_tokens in 3; do
            echo " Running scoring..."
            python3 metrics/run_mauve.py --input_file="$CKPTDIR/$DATASET/$fold/generated_${STRATEGY}_k=${max_test_aspect_tokens}.jsonl" \
                -a=$DATASET -i=$fold --model="maple" --max_samples=10000
            done
        done
    done
done


# run yelp and has to be higher
# dataset="yelp23"
# index=1
# CUDA_VISIBLE_DEVICES=2 python3 metrics/multiref_query_level_scorer.py -a="gest" -i=1 --model="maple" --input_file="/home/P76114511/projects/aspect_retriever/checkpoints/teacher_forcing_topk=1/gest/1/generated_supervised_k=2.json" \
#             -s="supervised" --max_samples=10000 --do_factuality --do_feature --do_diversity
# python3 metrics/multiref_query_level_scorer.py -a=$dataset -i=$index --model="pepler" --max_samples=1000

