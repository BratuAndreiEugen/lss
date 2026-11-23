import numpy as np
import soundfile as sf

# Parameters
sample_rate = 16000  # Hz
duration = 10         # seconds
noise_level = 0.005   # Noise amplitude

# Create an array of zeros
# audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

# Generate Gaussian white noise
audio = np.random.normal(
    0,
    noise_level,
    int(sample_rate * duration)
).astype(np.float32)

# Avoid clipping
audio = np.clip(audio, -1.0, 1.0)

# Save as WAV
sf.write("./test_silence/silence_with_noise.wav", audio, sample_rate)