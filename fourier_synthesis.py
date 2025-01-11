import librosa
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image
import soundfile as sf

def save_phase(audio: np.array, phase_path: str):
	stft = librosa.stft(audio)
	phase = np.angle(stft)
	np.save(phase_path, phase)
	print(f"Phase saved to {phase_path}")

def create_spectrogram_and_phase(audio_path, spectrogram_path, phase_path, n_mels=128, fmax=8000):
	# Load audio
	y, sr = librosa.load(audio_path, duration=5.0)

	# Get linear spectrogram and phase
	stft = librosa.stft(y)
	phase = np.angle(stft)
	np.save(phase_path, phase)

	# Create mel spectrogram
	mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
	mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

	# Normalize to 0-255 for image saving
	mel_normalized = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)

	# Save as image using PIL
	img = Image.fromarray(mel_normalized)
	img.save(spectrogram_path)

	return sr


def process_spectrogram(image_path):
	def frequency_scroll(img, shift=10):
		arr = np.array(img)
		arr_shifted = np.roll(arr, shift, axis=0)
		return Image.fromarray(arr_shifted)

	img = Image.open(image_path)
	img = frequency_scroll(img)
	spec_array = np.array(img.convert('L'))

	processed_img = Image.fromarray(spec_array)
	processed_img.save('processed_' + image_path)

	return spec_array


def synthesize_audio(processed_spec, phase_path, output_path, sr=22050):
	# Load phase
	phase = np.load(phase_path)

	# Denormalize spectrogram
	spec_denorm = (processed_spec / 255.0) * 80 - 40  # Approximate dB range

	# Convert to power scale
	spec_power = librosa.db_to_power(spec_denorm)

	# Convert mel to linear spectrogram
	linear_spec = librosa.feature.inverse.mel_to_stft(spec_power, sr=sr)

	# Combine with phase
	stft_reconstructed = linear_spec * np.exp(1j * phase)

	# Inverse STFT
	audio_reconstructed = librosa.istft(stft_reconstructed)

	# Save audio
	sf.write(output_path, audio_reconstructed, sr)

def plot_audio(audio_path, waveform_path):
	y, sr = librosa.load(audio_path, duration=5.0)
	plt.plot(y)
	plt.axis('off')
	plt.savefig(waveform_path, bbox_inches='tight', pad_inches=0, dpi=300)
	plt.close()

if __name__ == "__main__":
	audio_path = 'music.wav'
	spectrogram_path = 'spectrogram.png'
	phase_path = 'phase.npy'
	waveform_path = 'waveform.png'
	audio, sr = librosa.load(audio_path, duration=5.0)
	save_phase(audio, phase_path)
	sr = create_spectrogram_and_phase('music.wav', 'spectrogram.png', 'phase.npy')

	# Step 2: Edit spectrogram
	edited_spec = process_spectrogram('spectrogram.png')

	# Step 3: Synthesize audio
	synthesize_audio(edited_spec, 'phase.npy', 'reconstructed.wav', sr)
	plot_audio('reconstructed.wav', 'reconstructed.png')
