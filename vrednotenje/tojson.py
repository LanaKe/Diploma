import os
import json
import csv

json_file = "/shared/workspace/lrv/DeepBeauty/data/zalando/vitonhd_test_tagged.json"
# Read the JSON file
with open(json_file, 'r') as f:
    json_data = json.load(f)

# Handle if top-level JSON is a dict
if isinstance(json_data, dict):
    json_data = json_data.get("data", [])

annotation_list = [
    "colors",
    # "textures",
    "sleeveLength",
    "item",
]

prompts = []

# Collect annotation strings
for v in json_data:
    annotation_str = ""

    for tag in v.get("tag_info", []):
        if tag["tag_name"] in annotation_list and tag["tag_category"] is not None:
            annotation_str += tag["tag_category"] + " "

    cleaned_prompt = annotation_str.strip()
    if cleaned_prompt:  # Skip empty prompts
        prompts.append(cleaned_prompt)

# Save prompts to CSV (one per line)
print(f"Total samples in JSON: {len(json_data)}")
print(f"Prompts generated: {len(prompts)}")
missing_count = 2032 - len(prompts)
print(f"You need to add {missing_count} prompts.")

for _ in range(missing_count):
    prompts.append("high quality T-shirt image")

print(f"Prompts generated: {len(prompts)}")

csv_filename = "annotations_dodane.csv"
with open(csv_filename, mode="w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["prompt"])  # Header
    for prompt in prompts:
        writer.writerow([prompt])
