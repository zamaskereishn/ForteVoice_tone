#!/usr/bin/env python3
"""
Script to inspect and analyze corrupted audio samples in HuggingFace datasets.
"""

from datasets import load_dataset, Audio
import traceback

def inspect_dataset_audio(dataset_path, split="train", start_idx=87240, end_idx=87250):
    """
    Inspect specific audio samples to find corruption issues.
    """
    print(f"Loading dataset: {dataset_path}")
    
    try:
        split = 'train'
        dataset = load_dataset(dataset_path, split=split, token=True)
        dataset = dataset.cast_column('audio', Audio(16000, decode=False))
        start_idx = 0 
        end_idx = len(dataset)
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        
        # Focus on the problematic range
        print(f"\nInspecting samples {start_idx} to {end_idx}")
        
        for idx in range(start_idx, min(end_idx, len(dataset))):
            print(f"\n--- Sample {idx} ---")
            
            try:
                sample = dataset[idx]
                print(f"Sample keys: {list(sample.keys())}")
                
                # Check audio field
                if 'audio' in sample:
                    audio_info = sample['audio']
                    print(f"Audio type: {type(audio_info)}")
                    
                    if isinstance(audio_info, dict):
                        print(f"Audio keys: {list(audio_info.keys())}")
                        if 'path' in audio_info:
                            print(f"Audio path: {audio_info['path']}")
                        if 'bytes' in audio_info:
                            bytes_data = audio_info['bytes']
                            if bytes_data is not None:
                                print(f"Audio bytes length: {len(bytes_data)}")
                                print(f"First 20 bytes: {bytes_data[:20]}")
                            else:
                                print("Audio bytes is None!")
                    
                    # Try to decode the audio
                    print("Attempting to decode audio...")
                    try:
                        decoder = Audio(decode=True)
                        decoded = decoder.decode_example(audio_info)
                        print(f"✓ Audio decoded successfully")
                        print(f"  Array shape: {decoded['array'].shape}")
                        print(f"  Sample rate: {decoded['sampling_rate']}")
                    except Exception as decode_error:
                        print(f"✗ Audio decode failed: {decode_error}")
                        print(f"  Error type: {type(decode_error).__name__}")

                        with open('corrupted_audios.txt', 'w') as f:
                            f.write(f"Corrupted sample: {sample}\n")
                        
                        # Try to get more info about the corruption
                        if hasattr(audio_info, 'get') and 'bytes' in audio_info:
                            bytes_data = audio_info['bytes']
                            if bytes_data:
                                print(f"  Bytes analysis:")
                                print(f"    Length: {len(bytes_data)}")
                                print(f"    First 50 bytes: {bytes_data[:50]}")
                                print(f"    Last 50 bytes: {bytes_data[-50:]}")
                                
                                # Check for common audio file headers
                                if bytes_data[:4] == b'RIFF':
                                    print("    Detected: WAV file header")
                                elif bytes_data[:3] == b'ID3' or bytes_data[:2] == b'\xff\xfb':
                                    print("    Detected: MP3 file header")
                                elif bytes_data[:4] == b'fLaC':
                                    print("    Detected: FLAC file header")
                                else:
                                    print(f"    Unknown file format, header: {bytes_data[:10]}")
                
                # Check text field
                text_fields = ['text', 'sentence', 'transcription', 'transcript']
                for field in text_fields:
                    if field in sample:
                        text = sample[field]
                        print(f"Text ({field}): '{text[:100]}...' (length: {len(text)})")
                        break
                else:
                    print("No text field found")
                
            except Exception as sample_error:
                print(f"✗ Failed to load sample {idx}: {sample_error}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        traceback.print_exc()


def find_all_corrupted_samples(dataset_path, split="train", max_samples=1000):
    """
    Scan through dataset to find all corrupted audio samples.
    """
    print(f"Scanning dataset for corrupted audio samples...")
    
    try:
        split = 'train'
        # dataset = load_dataset(dataset_path, split=split, token=True)
        dataset = load_dataset(
            path=dataset_path,
            split=split,
            cache_dir=None,
            token=True,
            trust_remote_code=True,
        )

        print(f"Dataset loaded. Total samples: {len(dataset)}")

        dataset = dataset.cast_column('audio', Audio(16000, decode=False))
        
        corrupted_indices = []
        decoder = Audio(sampling_rate=16000, decode=True)
        
        # Scan through samples
        for idx in range(min(max_samples, len(dataset))):
            if idx % 1000 == 0:
                print(f"Scanned {idx} samples, found {len(corrupted_indices)} corrupted")
                
            try:
                sample = dataset[idx]
                audio_info = sample['audio']

                tmp_decoder = Audio(sampling_rate=16000, decode=True)
                y_raw = audio_info
                dec = tmp_decoder.decode_example(y_raw)
                y, y_sr = dec['array'], dec['sampling_rate']
                print(f"✓ Main script decoder success: array shape={y.shape}, sr={y_sr}")
            except Exception as e:
                print(f"Corrupted sample found at index {idx}: {e}")
                corrupted_indices.append(idx)
        
        print(f"\nScan complete. Found {len(corrupted_indices)} corrupted samples:")
        for idx in corrupted_indices:
            print(f"  - Sample {idx}")
        
        return corrupted_indices
        
    except Exception as e:
        print(f"Failed to scan dataset: {e}")
        return []


def save_corrupted_sample_info(dataset_path, corrupted_indices, output_file="corrupted_samples.txt"):
    """
    Save detailed info about corrupted samples to a file.
    """
    try:
        dataset = load_dataset(dataset_path, split="train", token=True)
        
        with open(output_file, 'w') as f:
            f.write(f"Corrupted samples in {dataset_path}\n")
            f.write(f"Total corrupted: {len(corrupted_indices)}\n\n")
            
            for idx in corrupted_indices:
                try:
                    sample = dataset[idx]
                    f.write(f"Sample {idx}:\n")
                    f.write(f"  Keys: {list(sample.keys())}\n")
                    
                    if 'audio' in sample:
                        audio_info = sample['audio']
                        if isinstance(audio_info, dict) and 'path' in audio_info:
                            f.write(f"  Audio path: {audio_info['path']}\n")
                        if isinstance(audio_info, dict) and 'bytes' in audio_info:
                            bytes_data = audio_info['bytes']
                            if bytes_data:
                                f.write(f"  Bytes length: {len(bytes_data)}\n")
                            else:
                                f.write(f"  Bytes: None\n")
                    
                    # Get text if available
                    text_fields = ['text', 'sentence', 'transcription']
                    for field in text_fields:
                        if field in sample:
                            f.write(f"  {field}: {sample[field][:100]}...\n")
                            break
                    
                    f.write("\n")
                    
                except Exception as e:
                    f.write(f"Sample {idx}: Error loading - {e}\n\n")
        
        print(f"Corrupted sample info saved to {output_file}")
        
    except Exception as e:
        print(f"Failed to save corrupted sample info: {e}")


if __name__ == "__main__":
    dataset_path = "farabi-lab/kazakh-stt"
    
    print("=== Audio Corruption Inspector ===\n")
    
    # Option 1: Inspect specific range around the error
    print("1. Inspecting samples around the error location...")
    inspect_dataset_audio(dataset_path, start_idx=0, end_idx=2)
    
    # Option 2: Find all corrupted samples (comment out if dataset is too large)
    # print("\n2. Scanning for all corrupted samples...")
    # corrupted = find_all_corrupted_samples(dataset_path, max_samples=5000)
    
    # Option 3: Save info about corrupted samples
    # if corrupted:
    #     save_corrupted_sample_info(dataset_path, corrupted)