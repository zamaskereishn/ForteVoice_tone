# from datasets import load_dataset, Audio

# cache_dir = "./hf_cache"

# dataset = load_dataset(
#     "farabi-lab/kazakh-stt",
#     split="train",
#     cache_dir=cache_dir,
#     streaming=True,  
# )

# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# for sample in dataset.take(3):
#     print(sample)
from datasets import load_dataset, Audio
import os, sys

# 1) Load (no streaming) — goes to ~/.cache/huggingface/datasets by default
print("Loading dataset metadata…")
ds = load_dataset("farabi-lab/kazakh-stt", split="train")

# 2) Keep decode lazy for prefetch (faster, no large arrays in RAM)
ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

# 3) Force all referenced audio files to be locally cached once
# Touching the "path" triggers download into cache without decoding
print("Prefetching audio into local HF cache… (first pass may take a while)")
def _touch(batch):
    _ = [a["path"] for a in batch["audio"]]
    return batch

ds.map(_touch, batched=True, batch_size=64, num_proc=8, desc="Prefetch")

print("Done. All audio should now be present in your HF datasets cache.")
