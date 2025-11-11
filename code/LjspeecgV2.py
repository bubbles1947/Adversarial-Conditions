import os
import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pyloudnorm as pyln
from tqdm import tqdm
import soundfile as sf

CONFIG = {
    'ljspeech_path': r"C:\Users\USER\Desktop\wavefake\LJSpeech-1.1\LJSpeech-1.1\wavs",
    'output_path': r"D:\processed_ljspeech",
    'sample_rate': 22050,
    'n_mels': 80,
    'n_fft': 2048,
    'hop_length': 512,
    'image_size': 299,
    'test_size': 0.1,
    'val_size': 0.1,
    'random_seed': 42
}

def preprocess_audio_file(audio_path, target_sr=22050, target_size=299, lufs_target=-23.0):
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)

        #trim silence
        intervals = librosa.effects.split(audio, top_db=40)
        audio_trimmed = np.concatenate([audio[start:end] for start, end in intervals])

        #LUFS normalization
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_trimmed)
        audio_norm = pyln.normalize.loudness(audio_trimmed, loudness, lufs_target)

        #prevent clipping
        audio_norm = np.clip(audio_norm, -1.0, 1.0)

        #mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_norm,
            sr=sr,
            n_mels=CONFIG['n_mels'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length']
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        #standardization
        mel_spec_std = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)

        #resize and 3-channel stack
        img = Image.fromarray(((mel_spec_std - mel_spec_std.min()) / (mel_spec_std.max() - mel_spec_std.min()) * 255).astype(np.uint8))
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        mel_3ch = np.stack([np.array(img_resized)/255.0]*3, axis=-1)

        return audio_norm, mel_3ch

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, None

def save_outputs(audio, mel, out_audio_path, out_mel_path, out_img_path):
    #save normalized audio
    sf.write(out_audio_path, audio, CONFIG['sample_rate'], subtype='PCM_16')

    #save mel spectrogram
    np.save(out_mel_path, mel.astype(np.float32))

    #save PNG with axes
    plt.figure(figsize=(4,4))
    plt.imshow(mel[:,:,0], origin='lower', aspect='auto', cmap='inferno')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Mel-Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_img_path)
    plt.close()


def preprocess_and_split_ljspeech():
    base_out = CONFIG['output_path']
    splits = ['train', 'val', 'test']

    for split in splits:
        for folder in ['audio', 'mel', 'images']:
            os.makedirs(os.path.join(base_out, split, folder), exist_ok=True)

    wav_files = sorted([os.path.join(CONFIG['ljspeech_path'], f)
                        for f in os.listdir(CONFIG['ljspeech_path']) if f.endswith('.wav')])

    #Train/Val/Test split
    train_val, test = train_test_split(wav_files, test_size=CONFIG['test_size'], random_state=CONFIG['random_seed'])
    train, val = train_test_split(train_val, test_size=CONFIG['val_size'], random_state=CONFIG['random_seed'])
    splits_dict = {'train': train, 'val': val, 'test': test}

    #process files
    for split_name, files in splits_dict.items():
        print(f"\nProcessing {split_name} set ({len(files)} files)")
        for fpath in tqdm(files, desc=split_name):
            fname = os.path.splitext(os.path.basename(fpath))[0]
            audio_out = os.path.join(base_out, split_name, 'audio', f"{fname}.wav")
            mel_out = os.path.join(base_out, split_name, 'mel', f"{fname}.npy")
            img_out = os.path.join(base_out, split_name, 'images', f"{fname}.png")

            audio_norm, mel_3ch = preprocess_audio_file(fpath, CONFIG['sample_rate'], CONFIG['image_size'])
            if audio_norm is not None:
                save_outputs(audio_norm, mel_3ch, audio_out, mel_out, img_out)

    print("\nLJSpeech preprocessing complete")

if __name__ == "__main__":
    preprocess_and_split_ljspeech()
