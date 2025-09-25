import json
import os 

# Read the manifest
manifest_path = "/home/askhat.sametov/Projects/kzASR/datasets/kazakh/farabi-lab/kazakh-stt/train/train_farabi-lab_kazakh-stt_manifest_clean.json"

# for file in os.listdir('/home/askhat.sametov/Projects/kzASR/datasets/kazakh/farabi-lab/kazakh-stt/train'): 
#     if file[-4:] == '.wav':
#         continue  
#     else: 
#         print(file)

cnt = 0 

with open(manifest_path, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        # Skip the corrupted sample
        if cnt == 1: 
            print(entry)

        cnt += 1 

print('total number of files: ', cnt) # 142969

# # Write cleaned manifest
# with open(manifest_path + '_cleaned.json', 'w') as f:
#     for entry in cleaned_entries:
#         f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# print(f"Removed corrupted entry. Original: {len(cleaned_entries)+1}, Cleaned: {len(cleaned_entries)}")
