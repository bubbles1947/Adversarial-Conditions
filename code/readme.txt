Preprocessing methods: 

Resampling → Convert all audio to a fixed rate (22.05 kHz or 16 kHz).

Silence trimming → Remove leading/trailing silence to reduce noise bias.

Loudness normalization (LUFS) → Keeps perceived volume consistent across files.

Mel-spectrogram scaling → Log-scaling (librosa.power_to_db) for perceptual similarity.

Standardization → Normalize each spectrogram to mean = 0, std = 1 for DNN stability.

