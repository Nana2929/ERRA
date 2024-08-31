export CUDA_VISIBLE_DEVICES=0
DATA_ROOT="./nete_format_data"
DATASET="yelp23"

for index in 1 2 3 4 5;do
    logpath="log/${DATASET}/${index}/run.log"
    # make dir
    mkdir -p log/${DATASET}/${index}
    echo "[ERRA] Training & inferencing index ${index}"

    python -u main.py \
    --auto_arg_by_dataset ${DATASET} \
    --index ${index} \
    --cuda \
    --epochs 200 \
    --checkpoint ./${DATASET}/${index} > ${logpath} 2>&1
done