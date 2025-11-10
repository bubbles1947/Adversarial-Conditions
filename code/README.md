#  1. Goal of preprocesing

Train a real-vs-fake audio classifier to detect synthetic speech from WaveFake datasets.

We use LJSpeech (English real audio) as the real class and only english waveFake folders as the fake class. [excluding the JUST dataset]. We preprocess all audio consistently (normalize, trim silence, generate Mel-spectrograms and images), then prepare the dataset for later transfer learning, data augmentation, and CNN training (ResNet-18, EfficientNet, MobileNet).

#  2. Dataset Overview
##  2.1 Real Audio

LJSpeech-1.1 (~3.5 GB)
English speech samples, .wav format (ljspeech-001-001.wav, etc.)
##  2.2 Fake Audio (WaveFake)

Root folder: D:\AOT wall\wavefake\generated_audio
This contains 10 subfolders, 2 of which are just* → Japanese → skipped. Remaining eight folders contain English synthetic audio:

ljspeech_full_band_melgan, ljspeech_hifigan, common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech and other English-generated folders. Each folder may have subfolders (e.g., generated/) → processed recursively. (.wav) files named like gen_0.wav, gen_1.wav.

#  3. Preprocessing Methodology

The preprocessing pipeline was designed to match LJSpeech preprocessing:

## 3.1 Steps for each audio file

1. Load .wav file at 22,050 Hz (SAMPLE_RATE = 22050).

2. Trim silence from start and end (librosa.effects.trim, top_db=20).

3. Normalize amplitude to [-1, 1] range (y / max(abs(y))).

4. Generate Mel-spectrogram:

N_MELS = 80, FRAME_SIZE = 2048, HOP_LENGTH = 512

5. Convert to dB (librosa.power_to_db)

6. Standardize ((mel - mean) / std)

7. Save outputs in structured folders:

Normalized .wav file → audio/

Mel-spectrogram .npy → mel/

Mel-spectrogram .png → images/
