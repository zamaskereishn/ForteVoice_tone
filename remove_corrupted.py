import json
import os 

# Read the manifest
manifest_path = "/home/askhat.sametov/Projects/kzASR/datasets/kazakh/farabi-lab/kazakh-stt/train/train_farabi-lab_kazakh-stt_manifest.json"
cleaned_entries = []
cnt = 0 

with open(manifest_path, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        # Skip the corrupted sample
        if entry.get('id') != 'dataset/audio4/69/1585_166':
            cleaned_entries.append(entry)
        else: 
            print("ENTRY: ", entry)

        cnt += 1 

print('total number of files: ', cnt) # 142969

# Write cleaned manifest
# with open(manifest_path + '_cleaned.json', 'w') as f:
#     for entry in cleaned_entries:
#         f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# print(f"Removed corrupted entry. Original: {len(cleaned_entries)+1}, Cleaned: {len(cleaned_entries)}")

# Create clean filename
base_name = os.path.splitext(manifest_path)[0]  # removes .json
clean_manifest_path = f"{base_name}_clean.json"

with open(clean_manifest_path, 'w') as f:
    for entry in cleaned_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Optionally replace the original
# os.replace(clean_manifest_path, manifest_path)

print(f"Created clean manifest: {clean_manifest_path}")
