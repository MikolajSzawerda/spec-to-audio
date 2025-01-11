import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import soundfile as sf


def save_phase(audio: np.array, phase_path: str):
	stft = librosa.stft(audio)
	phase = np.angle(stft)
	np.save(phase_path, phase)
	print(f"Phase saved to {phase_path}")


def create_spectrogram(audio: np.array, sr: int | float, spectrogram_path: str, n_mels=128, fmax=8000):
	mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax)
	mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

	# Normalize to 0-255 for image saving
	mel_normalized = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)

	# Save as image using PIL
	img = Image.fromarray(mel_normalized)
	img.save(spectrogram_path)

	return sr


def process_spectrogram(img: Image):
	def frequency_scroll(img, shift=10):
		arr = np.array(img)
		arr_shifted = np.roll(arr, shift, axis=0)
		return Image.fromarray(arr_shifted)

	# img = frequency_scroll(img)
	spec_array = np.array(img.convert('L'))

	return spec_array


def synthesize_audio(processed_spec: np.array, phase: np.array, output_audio_path, sr=22050):
	spec_denorm = (processed_spec / 255.0) * 80 - 40  # Approximate dB range
	spec_power = librosa.db_to_power(spec_denorm)
	linear_spec = librosa.feature.inverse.mel_to_stft(spec_power, sr=sr)
	stft_reconstructed = linear_spec * np.exp(1j * phase)
	audio_reconstructed = librosa.istft(stft_reconstructed)
	sf.write(output_audio_path, audio_reconstructed, sr)


def plot_audio(audio_path, waveform_path):
	y, sr = librosa.load(audio_path, duration=5.0)
	plt.plot(y)
	plt.axis('off')
	plt.savefig(waveform_path, bbox_inches='tight', pad_inches=0, dpi=300)
	plt.close()


audio_path = 'music.wav'
spectrogram_path = 'spectrogram.png'
phase_path = 'phase.npy'
waveform_path = 'waveform.png'
if __name__ == "__main__":
	audio, sr = librosa.load(audio_path, duration=5.0, offset=5.0)
	save_phase(audio, phase_path)
	create_spectrogram(audio, sr, spectrogram_path)

	spectrogram_image = Image.open(spectrogram_path)
	edited_spec = process_spectrogram(spectrogram_image)

	processed_img = Image.fromarray(edited_spec)
	processed_img.save('processed_' + spectrogram_path)

	phase = np.load(phase_path)
	synthesize_audio(edited_spec, phase, 'reconstructed.wav', sr)
	plot_audio('reconstructed.wav', 'reconstructed.png')
