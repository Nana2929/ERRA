for index in 2 3 4 5;do
    echo "Processing index ${index}"
    echo "Retrieving data"
    python3 retrieve_process.py --index ${index}
    # echo "getting aspect"
    # python3 aspect_process.py --index ${index}
    done