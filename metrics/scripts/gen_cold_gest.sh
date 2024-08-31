for index in {1..5};do
    python3 /home/P76114511/projects/aspect_retriever/metrics/remove_cold_ids.py -a=gest -i=$index --model="maple"
    done

# index=1
# suffix: dbloss_no_merged
# len(warm_users): 35228
# len(warm_items): 29115
# The original len(test_data): 3751
# The new len(test_data): 1668

# index=2
# suffix: dbloss_no_merged
# len(warm_users): 35298
# len(warm_items): 29136
# The original len(test_data): 3723
# The new len(test_data): 1692

# index=3
# suffix: dbloss_no_merged
# len(warm_users): 35322
# len(warm_items): 29144
# The original len(test_data): 3696
# The new len(test_data): 1700

# index=4
# suffix: dbloss_no_merged
# len(warm_users): 35262
# len(warm_items): 29124
# The original len(test_data): 3731
# The new len(test_data): 1672

# index=5
# suffix: dbloss_no_merged
# len(warm_users): 35262
# len(warm_items): 29124
# The original len(test_data): 3717
# The new len(test_data): 1668