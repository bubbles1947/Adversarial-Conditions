#  1. Goal of preprocesing

Train a real-vs-fake audio classifier to detect synthetic speech from WaveFake datasets.

Use LJSpeech (English real audio) as the real class.

Use English WaveFake folders as the fake class.

Preprocess all audio consistently (normalize, trim silence, generate Mel-spectrograms and images).

Prepare the dataset for later transfer learning, data augmentation, and CNN training (ResNet-18, EfficientNet, MobileNet).

#  2. Dataset Overview
##  2.1 Real Audio

LJSpeech-1.1 (~3.5 GB)

English speech samples, .wav format (ljspeech-001-001.wav, etc.)

##  2.2 Fake Audio (WaveFake)

Root folder: D:\AOT wall\wavefake\generated_audio

Contains 10 subfolders, 2 of which are just* → Japanese → skipped

Remaining eight folders contain English synthetic audio:

ljspeech_full_band_melgan

ljspeech_hifigan

common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech

Other English-generated folders. Each folder may have subfolders (e.g., generated/) → processed recursively. (.wav) files named like gen_0.wav, gen_1.wav.

#  3. Preprocessing Methodology

The preprocessing pipeline was designed to match LJSpeech preprocessing:

## 3.1 Steps for each audio file

Load .wav file at 22,050 Hz (SAMPLE_RATE = 22050).

Trim silence from start and end (librosa.effects.trim, top_db=20).

Normalize amplitude to [-1, 1] range (y / max(abs(y))).

Generate Mel-spectrogram:

N_MELS = 80, FRAME_SIZE = 2048, HOP_LENGTH = 512

Convert to dB (librosa.power_to_db)

Standardize ((mel - mean) / std)

Save outputs in structured folders:

Normalized .wav file → audio/

Mel-spectrogram .npy → mel/

Mel-spectrogram .png → images/
