set -e
for index in 1 2 3;do
    echo "Processing index ${index}"
    echo "Retrieving data"
    python3 retrieve_process.py --index ${index} --dataset "yelp23"
    # echo "getting aspect"
    # python3 aspect_process.py --index ${index}
done