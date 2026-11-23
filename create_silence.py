import numpy as np
import soundfile as sf

# Parameters
sample_rate = 16000  # Hz
duration = 10         # seconds

# Create an array of zeros
audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

# Save as WAV
sf.write("./test_silence/silence.wav", audio, sample_rate)