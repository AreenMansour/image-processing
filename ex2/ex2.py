import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft
import soundfile as sf


def find_max_frequency_index(data) -> int:
    # Find the index of the maximum frequency magnitude
    max_freq_index = np.argmax(np.abs(np.fft.fft(data)))
    return max_freq_index


def q1(audio_path) -> np.array:
    # Read audio file
    fs, data = wavfile.read(audio_path)

    # Find the index of the maximum frequency magnitude
    max_freq_index = find_max_frequency_index(data)

    # Remove the frequency and its conjugate symmetric counterpart
    data_fft = np.fft.fft(data)
    data_fft[max_freq_index] = 0
    data_fft[-max_freq_index] = 0

    # Inverse FFT to obtain denoised audio
    y_denoised = np.fft.ifft(data_fft).real

    return y_denoised


def q2(file_path) -> np.array:
    # Load the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Compute the STFT using scipy.signal.stft
    _, _, stft_result = stft(data, fs=sample_rate, nperseg=1000)  # Adjust nperseg as needed

    # Convert the result to a numpy array (magnitude and phase)
    stft_magnitude = np.abs(stft_result)
    stft_phase = np.angle(stft_result)

    # Define the frequency bins corresponding to the STFT
    freqs = np.fft.fftfreq(stft_magnitude.shape[0], d=1 / sample_rate)

    # Find the indices corresponding to the specified frequency range
    start_frequency = 1100
    end_frequency = 1250
    start_index = np.argmax(freqs >= start_frequency)
    end_index = np.argmax(freqs >= end_frequency)

    # Apply the band-reject filter to the specified time frames
    filtered_stft_magnitude = np.copy(stft_magnitude)
    filtered_stft_magnitude[start_index:end_index, :] = 0

    # Reconstruct the time-domain signal using scipy.signal.istft
    _, filtered_data = istft(filtered_stft_magnitude * np.exp(1j * stft_phase),
                              fs=sample_rate, nperseg=1000)  # Adjust nperseg as needed

    # Save the filtered audio to a new file (optional)
    sf.write('filtered_audio.wav', filtered_data, sample_rate)

    return filtered_data

# Example usage for q1:
# file_path_q1 = '/cs/usr/areen0507/Downloads/q1.wav'
# q1_denoised, fs_q1 = q1(file_path_q1)
#
# # Plot the denoised spectrogram for q1
# plt.specgram(q1_denoised, Fs=fs_q1, cmap='viridis')
# plt.title('Q1 Denoised Spectrogram')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.show()
#
# # Example usage for q2:
# file_path_q2 = '/cs/usr/areen0507/Downloads/q2.wav'
# q2_denoised = q2(file_path_q2)
#
# # Plot the spectrogram for q2
# plt.specgram(q2_denoised, Fs=fs_q1, cmap='viridis')  # Using the same fs_q1 for consistency
# plt.title('Q2 Denoised Spectrogram')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.show()
