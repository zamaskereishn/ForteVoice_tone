from datasets import Audio, load_dataset
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import TrainingArguments
from transformers import Trainer

import os
import gc
import re
import json 
import numpy as np

from tone.training.data_collator import DataCollatorCTCWithPadding
from tone.training.model_wrapper import ToneForCTC

import torch
import torch.nn as nn
import torchaudio

import evaluate

from datasets import disable_caching

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Sample rate used in the base model
SAMPLE_RATE = 8000

# Padding added to each audio during training to improve performance
LEFT_PAD_MS = 300
RIGHT_PAD_MS = 300
LEFT_PAD_SIZE = round(SAMPLE_RATE * LEFT_PAD_MS * 1e-3)
RIGHT_PAD_SIZE = round(SAMPLE_RATE * RIGHT_PAD_MS * 1e-3)

def load_from_local(
    train_path,
    validation_path,
    sample_rate=SAMPLE_RATE,
    audio_column="audio",
    text_column="text",
    ):
    """
    Load audios from a JSON-lines manifest containing the following features in each row: audio and text.
    """
    manifest_dict = {"train": train_path, "validation": validation_path}

    print("OUTPUT FROM load_from_local: ", train_path, validation_path)

    dataset = load_dataset("json", data_files=manifest_dict)
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if audio_column != "audio":
        dataset = dataset.rename_column(audio_column, "audio")
    dataset = dataset.cast_column('audio', Audio(sampling_rate=sample_rate))
    return dataset


def load_from_huggingface(
        dataset_name,
        sample_rate=SAMPLE_RATE,
        audio_column="audio",
        text_column="text",
        **kwargs
    ):
    """
    Load an ASR dataset from huggingface. 
    You might need to change this function if you use a custom dataset.
    """
    dataset = load_dataset(dataset_name, **kwargs)
    dataset = dataset.select_columns([audio_column, text_column])
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if audio_column != "audio":
        dataset = dataset.rename_column(audio_column, "audio")
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))
    return dataset

dataset = load_from_local('datasets/kazakh/farabi-lab/kazakh-stt/train/train_farabi-lab_kazakh-stt_manifest.json', 'datasets/kazakh/farabi-lab/kazakh-stt/val/val_farabi-lab_kazakh-stt_manifest.json', audio_column='audio_filepath')
# dataset = load_from_local('datasets/kazakh/farabi-lab/kazakh-stt/train[:5]/train[:5]_farabi-lab_kazakh-stt_manifest.json', 'datasets/kazakh/farabi-lab/kazakh-stt/train[:5]/train[:5]_farabi-lab_kazakh-stt_manifest.json', audio_column='audio_filepath')

# disable_caching()  

# dataset["train"] = dataset["train"].select(range(10))
# dataset["validation"] = dataset["validation"].select(range(5))
print(dataset)

# audio_data = dataset['train'][0]['audio']

# print(f"Audio array shape: {audio_data['array'].shape}")
# print(f"Sampling rate: {audio_data['sampling_rate']}")
# print(f"Audio array (first 10 samples): {audio_data['array'][:10]}")

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "t-tech/T-one", pad_token="[PAD]", word_delimiter_token="|"
)

# Add Kazakh-specific characters
kazakh_letters = ['ә', 'ғ', 'қ', 'ң', 'ө', 'ұ', 'ү', 'һ', 'і']
num_added_tokens = tokenizer.add_tokens(kazakh_letters)
print(f"Added {num_added_tokens} new tokens")

# Save the extended tokenizer
tokenizer.save_pretrained("./kazakh-extended-tokenizer")

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=SAMPLE_RATE,
    padding_value=0.0,
    return_attention_mask=False,
    do_normalize=False,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# How many processors to use for data processing
# NUM_PROC = os.cpu_count()
NUM_PROC = 1
print(NUM_PROC)

REG = re.compile("[а-яёәғқңөұүһі]+")  # Added: ә ғ қ ң ө ұ ү һ і

# Global counter for cache clearing
_sample_counter = 0

def prepare_dataset(batch):
    global _sample_counter
    _sample_counter += 1

    audio = batch["audio"]
    audio_array = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    # Add magic padding on both sides to improve model performance
    batch["input_values"] = np.pad(audio_array, (LEFT_PAD_SIZE, RIGHT_PAD_SIZE), mode="constant")
    batch["input_lengths"] = len(batch["input_values"])
    
    text = " ".join(REG.findall(batch["text"].lower()))
    batch["labels"] = processor(text=text).input_ids

    # Clear cache every 5000 samples
    if _sample_counter % 5000 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Cache cleared at sample {_sample_counter}")

    return batch

_sample_counter = 0

dataset = dataset.map(prepare_dataset, remove_columns=["audio", "text"], num_proc=NUM_PROC)

gc.collect()

max_input_length_in_sec = 20.0
dataset["train"] = dataset["train"].filter(
    lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_lengths"]
)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToneForCTC.from_pretrained("t-tech/T-one").to(device)

decoder = model.tone.decoder.decoder_layers[0]
new_vocab_size = len(tokenizer)

new_decoder = nn.Conv1d(
    in_channels=384,
    out_channels=new_vocab_size, 
    kernel_size=1,
    stride=1
).to(device)

# Copy existing weights for original vocabulary
with torch.no_grad():
    new_decoder.weight[:35] = decoder.weight
    if decoder.bias is not None:
        new_decoder.bias[:35] = decoder.bias

model.tone.decoder.decoder_layers[0] = new_decoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

trainable_params = count_parameters(model)
print(f"Trainable parameters before freezing: {trainable_params:,}")


for param in model.tone.preprocessor.parameters():
    param.requires_grad = False

for param in model.tone.encoder.pre_encode.parameters():
    param.requires_grad = False

for i in range(7): 
    for param in model.tone.encoder.layers[i].parameters():
        param.requires_grad = False

trainable_params = count_parameters(model)
print(f"Trainable parameters after freezing: {trainable_params:,}")

wer_metric = evaluate.load("wer")


def compute_metrics(preds):
    pred_logits = preds.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Pred ids get padded by -100
    pred_ids[(pred_logits == -100).all(axis=-1)] = processor.tokenizer.pad_token_id
    preds.label_ids[preds.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Group repeating tokens to get the final transcription
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(preds.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# After loading model and before training
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


training_args = TrainingArguments(
    output_dir = "tone_ctc_train",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    # dataloader_num_workers = 2,
    # eval_on_start = True,
    num_train_epochs = 50,
    bf16 = True,
    lr_scheduler_type = "linear",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    logging_strategy = "epoch",
    learning_rate = 5e-5,
    weight_decay = 1e-6,
    warmup_ratio = 0.05,
    save_total_limit = 3,
    # torch_empty_cache_steps = 1000,
    # resume_from_checkpoint = "/home/askhat.sametov/Projects/kzASR/tone_ctc/checkpoint-19135",
    gradient_accumulation_steps = 4,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
)

print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

trainer.train()