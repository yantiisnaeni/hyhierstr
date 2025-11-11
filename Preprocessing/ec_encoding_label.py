import pandas as pd
import os

BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = "your_data_dir"  # input data training directory

# real-world dataset preprocessing result
ec_path = os.path.join(DATA_DIR,"ec_dataset_cleaned.csv")
ec_df = pd.read_csv(ec_path)

# label mapping file
mapping_path = os.path.join(DATA_DIR, "unique_label_levels_mapping.csv")
mapping_df = pd.read_csv(mapping_path)


# Data Structure
print("ec_df column:", ec_df.columns.tolist())
print("mapping_df column:", mapping_df.columns.tolist())

# ensure the column that used for merging
merge_key_ec = "kbli_code"
merge_key_map = "label_level_5"  

# MERGE ec_df with mapping_df
merged_df = pd.merge(
    ec_df,
    mapping_df,
    left_on=merge_key_ec,
    right_on=merge_key_map,
    how="left"
)

print(f"Merge done. Row number: {len(merged_df)}")

# Rename encoded → label level
rename_map = {
    "label_level_1_encoded": "label_level_1",
    "label_level_2_encoded": "label_level_2",
    "label_level_3_encoded": "label_level_3",
    "label_level_4_encoded": "label_level_4",
    "label_level_5_encoded": "label_level_5"
}
merged_df = merged_df.rename(columns=rename_map)

# Filtering Column
cols_keep = [
    "cleaned_text",
    "label_level_1",
    "label_level_2",
    "label_level_3",
    "label_level_4",
    "label_level_5"
]

final_df = merged_df[cols_keep]

# Save Results
output_path = os.path.join(DATA_DIR, "your_final_real_world_dataset.csv")
final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Final dataset: {output_path}")
print(final_df.head(5))
