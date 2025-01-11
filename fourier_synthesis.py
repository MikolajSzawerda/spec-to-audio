import librosa
import numpy as np
import skimage.io


def save_spectrogram_png(
		audio_path,
		out_png="spectrogram.png",
		n_fft=1024,
		hop_length=512,
		db_min=-80.0
):
	y, sr = librosa.load(audio_path, sr=None)
	S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
	magnitude, phase = librosa.magphase(S)
	mag_db = librosa.amplitude_to_db(magnitude, ref=1.0)
	mag_db_clipped = np.clip(mag_db, a_min=db_min, a_max=0)
	mag_db_norm = (mag_db_clipped - db_min) / float(-db_min)
	magnitude_8bit = (mag_db_norm * 255.0).astype(np.uint8)
	skimage.io.imsave(out_png, magnitude_8bit)
	return sr, S.shape, 0


def load_spectrogram_and_reconstruct(
		png_path,
		sr,
		original_stft_shape,
		mag_max,
		n_fft=1024,
		hop_length=512,
		db_min=-80.0
):
	magnitude_8bit = skimage.io.imread(png_path, as_gray=True)
	mag = magnitude_8bit.astype(np.float32) / 255.0
	mag_db_clipped = mag * -db_min + db_min

	magnitude_linear = np.power(10.0, mag_db_clipped / 20.0)
	return librosa.griffinlim(
		magnitude_linear,
		n_iter=32,
		hop_length=hop_length,
		win_length=n_fft
	)


if __name__ == "__main__":
	import soundfile as sf
	N_FFT = 1024
	sr, stft_shape, mag_max = save_spectrogram_png(
		audio_path="music.wav",
		out_png="spectrogram.png",
		n_fft=N_FFT,
		hop_length=512
	)

	y_reconstructed = load_spectrogram_and_reconstruct(
		png_path="spectrogram.png",
		sr=sr,
		original_stft_shape=stft_shape,
		mag_max=mag_max,
		n_fft=N_FFT,
		hop_length=512
	)
	sf.write("reconstructed.wav", y_reconstructed, sr)
