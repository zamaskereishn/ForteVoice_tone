import torch
from datasets import Audio, load_dataset
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import os
import re
import numpy as np
from tone.training.data_collator import DataCollatorCTCWithPadding
from tone.training.model_wrapper import ToneForCTC
import torch.nn as nn
import torchaudio 

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

dataset = load_from_local('datasets/kazakh/farabi-lab/kazakh-stt/test/test_farabi-lab_kazakh-stt_manifest.json', 'datasets/kazakh/farabi-lab/kazakh-stt/val/val_farabi-lab_kazakh-stt_manifest.json', audio_column='audio_filepath')

# print(dataset)

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
NUM_PROC = os.cpu_count()

print(NUM_PROC)
REG = re.compile("[а-яёәғқңөұүһі]+")  # Added: ә ғ қ ң ө ұ ү һ і

def prepare_dataset(batch):
    audio = batch["audio"]
    audio_array = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    # Add magic padding on both sides to improve model performance
    batch["input_values"] = np.pad(audio_array, (LEFT_PAD_SIZE, RIGHT_PAD_SIZE), mode="constant")
    batch["input_lengths"] = len(batch["input_values"])
    
    text = " ".join(REG.findall(batch["text"].lower()))
    batch["labels"] = processor(text=text).input_ids
    return batch


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

print("Model structure:")

with open('tone_model_updated.txt', 'w', encoding='utf-8') as f: 
    for name, module in model.named_modules():
        # f.write(f"Linear layer '{name}': {module.in_features} -> {module.out_features}\n")
        f.write(f"module {name}: {module}\n")

model.eval()

def transcribe_audio(audio_path, model, processor, device):
    """
    Transcribe a single audio file
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed (your model expects 8kHz)
    if sample_rate != 8000:
        resampler = torchaudio.transforms.Resample(sample_rate, 8000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Process audio with your processor
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=8000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    
    original_length = input_values.shape[1]

    padded_input = torch.nn.functional.pad(
        input_values, 
        (LEFT_PAD_SIZE, RIGHT_PAD_SIZE), 
        mode='constant', 
        value=0
    )
    
    input_lengths = torch.tensor([original_length], dtype=torch.long).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_values=padded_input, input_lengths=input_lengths)

        with open('tone_model_updated.txt', 'w', encoding='utf-8') as f: 
            for name, module in model.named_modules():
                # f.write(f"Linear layer '{name}': {module.in_features} -> {module.out_features}\n")
                f.write(f"module {name}: {module}\n")

        logits = outputs.logits
    
    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids.squeeze().cpu().numpy())
    
    return transcription

audio_file = "/home/askhat.sametov/Projects/tts/ru_audio.wav"
transcription_1 = transcribe_audio(audio_file, model, processor, device)
print(f"Transcription: {transcription_1}")

audio_file = "/home/askhat.sametov/Projects/tts/ru_audio3.wav"
transcription_2 = transcribe_audio(audio_file, model, processor, device)
print(f"Transcription: {transcription_2}")

with open('decoded_transcriptions.txt', 'w', encoding='utf-8') as f: 
    f.write(transcription_1)
    f.write('\n')
    f.write(transcription_2)

