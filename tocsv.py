import os
import csv

# Replace these with your folder paths
folder_paths = {
    'cloth': '/shared/workspace/lrv/DeepBeauty/data/zalando/train/cloth',
    'pose': '/shared/home/lana.kejzar/Diploma/train_op15_pose',
    'target': '/shared/home/lana.kejzar/Diploma/SAM/izluscena_oblacila',
    'mask': '/shared/workspace/lrv/DeepBeauty/data/zalando/train/gt_cloth_warped_mask'
}

# Collect filenames from each folder
all_filenames = {}
for folder_name, path in folder_paths.items():
    filenames = sorted(os.listdir(path))  # Sort to maintain order
    all_filenames[folder_name] = filenames


# Create a CSV file
output_csv = 'trening_maske.csv'

# Find the maximum length of the filenames lists
max_length = max(len(filenames) for filenames in all_filenames)

# Pad lists to match the max length
padded_filenames = {
    folder_name: filenames + [''] * (max_length - len(filenames))
    for folder_name, filenames in all_filenames.items()
}

# Write to the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header row with folder names
    writer.writerow(folder_paths.keys())
    # Write filenames row by row
    for row in zip(*padded_filenames.values()):
        writer.writerow(row)

print(f"CSV file '{output_csv}' created successfully.")
