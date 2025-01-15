import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
from matplotlib.widgets import Button, RadioButtons, Slider
import random

SOUND_DURATION_LIMIT_SECONDS = 5
TOP_DB = 80


def save_phase(audio: np.array, phase_path: str):
    stft = librosa.stft(audio)
    phase = np.angle(stft)
    np.save(phase_path, phase)
    print(f"Phase saved to {phase_path}")


def create_spectrogram(audio: np.array, sr: int, spectrogram_path: str, n_mels=128, fmax=8000):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=TOP_DB)
    mel_normalized = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)
    img = Image.fromarray(mel_normalized)
    img.save(spectrogram_path)


def process_spectrogram(img: Image, method: str, shift: int | None = 10, rect_params: dict = None):
    def frequency_scroll(img, shift=shift):
        arr = np.array(img)
        arr_shifted = np.roll(arr, shift, axis=0)
        return Image.fromarray(arr_shifted)

    def flip_time_axis(img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def cut_random_frequencies(img, rect_params):
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        top_left_x = rect_params['x_shift']
        top_left_y = rect_params['y_shift']
        rect_width = rect_params['width']
        rect_height = rect_params['height']
        bottom_right_x = min(top_left_x + rect_width, img_width)
        bottom_right_y = min(top_left_y + rect_height, img_height)
        draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=0)
        return img

    if method == 'Frequency Scroll':
        img = frequency_scroll(img)
    elif method == 'Flip Time Axis':
        img = flip_time_axis(img)
    elif method == 'Cut Random Frequencies':
        img = cut_random_frequencies(img, rect_params)

    spec_array = np.array(img.convert('L'))
    return spec_array


def synthesize_audio(processed_spec: np.array, phase: np.array, output_audio_path, sr=22050):
    spec_denorm = (processed_spec / 255.0) * TOP_DB - TOP_DB / 2
    spec_power = librosa.db_to_power(spec_denorm)
    linear_spec = librosa.feature.inverse.mel_to_stft(spec_power, sr=sr)
    stft_reconstructed = linear_spec * np.exp(1j * phase)
    audio_reconstructed = librosa.istft(stft_reconstructed)
    sf.write(output_audio_path, audio_reconstructed, sr)


def plot_audio_and_buttons(original_audio_path, sr, original_spectrogram_path, phase_path, processed_audio_path,
                           processed_spectrogram_path):
    y, _ = librosa.load(original_audio_path, sr=sr, duration=5.0)
    create_spectrogram(y, sr, original_spectrogram_path)

    fig, axs = plt.subplots(4, 1, figsize=(14, 14), num="Fourier Synthesis")
    plt.subplots_adjust(bottom=0.35)

    ax_original_waveform = axs[0]
    ax_original_waveform.plot(y)
    ax_original_waveform.set_title('Original Waveform')
    ax_original_waveform.axis('off')

    ax_original_spectrogram = axs[1]
    original_spec_img = Image.open(original_spectrogram_path)
    ax_original_spectrogram.imshow(original_spec_img, aspect='auto')
    ax_original_spectrogram.set_title('Original Spectrogram')
    ax_original_spectrogram.axis('off')

    ax_reconstructed_waveform = axs[2]
    ax_reconstructed_waveform.set_title('Reconstructed Waveform')
    ax_reconstructed_waveform.axis('off')

    ax_reconstructed_spectrogram = axs[3]
    ax_reconstructed_spectrogram.set_title('Reconstructed Spectrogram')
    ax_reconstructed_spectrogram.axis('off')

    ax_radio = plt.axes([0.05, 0.15, 0.25, 0.1], facecolor=(0.95, 0.95, 0.95))
    radio = RadioButtons(ax_radio, ('Frequency Scroll', 'Flip Time Axis', 'Cut Random Frequencies'))
    radio.set_active(0)

    ax_slider = plt.axes([0.35, 0.15, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Shift', valmin=1, valmax=100, valinit=10, valstep=1)

    ax_width_slider = plt.axes([0.35, 0.24, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    width_slider = Slider(ax_width_slider, 'Width', valmin=10, valmax=128, valinit=80, valstep=1)
    ax_height_slider = plt.axes([0.35, 0.21, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    height_slider = Slider(ax_height_slider, 'Height', valmin=10, valmax=128, valinit=10, valstep=1)
    ax_xshift_slider = plt.axes([0.35, 0.18, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    xshift_slider = Slider(ax_xshift_slider, 'X', valmin=0, valmax=130, valinit=0, valstep=1)
    ax_yshift_slider = plt.axes([0.35, 0.15, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    yshift_slider = Slider(ax_yshift_slider, 'Y', valmin=0, valmax=130, valinit=0, valstep=1)

    ax_slider.set_visible(True)
    ax_width_slider.set_visible(False)
    ax_height_slider.set_visible(False)
    ax_xshift_slider.set_visible(False)
    ax_yshift_slider.set_visible(False)

    ax_process_button = plt.axes([0.05, 0.05, 0.2, 0.075])
    btn_process = Button(ax_process_button, 'Process Audio')

    ax_play_original_button = plt.axes([0.54, 0.05, 0.2, 0.075])
    btn_play_original = Button(ax_play_original_button, 'Play Original Audio')

    ax_play_button = plt.axes([0.76, 0.05, 0.2, 0.075])
    btn_play = Button(ax_play_button, 'Play Reconstructed Audio')
    btn_play.active = False

    def toggle_sliders(label):
        is_cut_random = (label == 'Cut Random Frequencies')
        ax_slider.set_visible(label == 'Frequency Scroll')
        ax_width_slider.set_visible(is_cut_random)
        ax_height_slider.set_visible(is_cut_random)
        ax_xshift_slider.set_visible(is_cut_random)
        ax_yshift_slider.set_visible(is_cut_random)
        fig.canvas.draw_idle()

    def process_audio(event):
        method = radio.value_selected
        shift_value = slider.val if method == 'Frequency Scroll' else None
        rect_params = {
            'width': width_slider.val,
            'height': height_slider.val,
            'x_shift': xshift_slider.val,
            'y_shift': yshift_slider.val
        }

        spectrogram_image = Image.open(original_spectrogram_path)
        edited_spec = process_spectrogram(spectrogram_image, method, shift=shift_value, rect_params=rect_params)
        processed_img = Image.fromarray(edited_spec)
        processed_img.save(processed_spectrogram_path)

        phase = np.load(phase_path)
        synthesize_audio(edited_spec, phase, processed_audio_path, sr)

        y_reconstructed, _ = librosa.load(processed_audio_path, sr=sr, duration=SOUND_DURATION_LIMIT_SECONDS)
        ax_reconstructed_waveform.clear()
        ax_reconstructed_waveform.plot(y_reconstructed)
        ax_reconstructed_waveform.set_title('Reconstructed Waveform')
        ax_reconstructed_waveform.axis('off')

        reconstructed_spec_img = Image.open(processed_spectrogram_path)
        ax_reconstructed_spectrogram.clear()
        ax_reconstructed_spectrogram.imshow(reconstructed_spec_img, aspect='auto')
        ax_reconstructed_spectrogram.set_title('Reconstructed Spectrogram')
        ax_reconstructed_spectrogram.axis('off')
        fig.canvas.draw_idle()

        btn_play.active = True

    def play_original_audio(event):
        print("Playing original audio...")
        data, samplerate = sf.read(original_audio_path)
        samples_to_play = int(SOUND_DURATION_LIMIT_SECONDS * samplerate)
        short_data = data[:samples_to_play]
        sd.play(short_data, samplerate)
        sd.wait()

    def play_audio(event):
        if btn_play.active:
            print("Playing reconstructed audio...")
            data, samplerate = sf.read(processed_audio_path)
            samples_to_play = int(SOUND_DURATION_LIMIT_SECONDS * samplerate)
            short_data = data[:samples_to_play]
            sd.play(short_data, samplerate)
            sd.wait()

    radio.on_clicked(toggle_sliders)
    btn_process.on_clicked(process_audio)
    btn_play_original.on_clicked(play_original_audio)
    btn_play.on_clicked(play_audio)

    plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    original_audio_path = filedialog.askopenfilename(title="Select an audio file", filetypes=[("WAV files", "*.wav")])
    if original_audio_path:
        original_spectrogram_path = 'original_spectrogram.png'
        phase_path = 'phase.npy'
        processed_audio_path = 'reconstructed.wav'
        processed_spectrogram_path = 'reconstructed_spectrogram.png'

        audio, sr = librosa.load(original_audio_path, duration=SOUND_DURATION_LIMIT_SECONDS)
        save_phase(audio, phase_path)

        plot_audio_and_buttons(original_audio_path, sr, original_spectrogram_path, phase_path, processed_audio_path,
                               processed_spectrogram_path)