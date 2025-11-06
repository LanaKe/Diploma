import os
import json
import csv

import json

json_file = "/shared/workspace/lrv/DeepBeauty/data/zalando/vitonhd_test_tagged.json"
new_file = "/shared/home/lana.kejzar/Diploma/vrednotenje/test3.txt"

# Read the original JSON file
with open(json_file, 'r') as f:
    json_data = json.load(f)

# Write the same data to a new file
with open(new_file, 'w') as f:
    json.dump(json_data, f, indent=4)  # indent=4 for pretty formatting


