from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

# ========================= CONFIG =========================
image_path = r"chart.png"     # <<< CHANGE THIS

sampling_rate = 4000                           # <<< Adjust based on your signal
window_size = 250                              # Samples per window (adjust as needed)
window_interval = 200                          # Step size between windows (set = window_size for non-overlapping)
padding_factor = 4                             # 4x zero padding

show_time_domain = True
max_freq = 800                                 # Limit frequency axis (low range focus)
use_log_scale = True                         # Set True for logarithmic frequency axis
# ========================================================

# Load and extract waveform
img = Image.open(image_path).convert('RGB')
img_array = np.array(img)

r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
mask = (r > 180) & (g < 100) & (b < 100)

height, width = mask.shape[:2]
y_values = np.zeros(width)

for x in range(width):
    column = mask[:, x]
    if np.any(column):
        y_values[x] = np.mean(np.where(column)[0])
    else:
        y_values[x] = np.nan

y_values = np.interp(np.arange(width), 
                     np.where(~np.isnan(y_values))[0], 
                     y_values[~np.isnan(y_values)])

y = height - y_values
y = y - np.mean(y)
y = y / np.max(np.abs(y))          # Normalize

N = len(y)
print(f"Extracted {N} samples")

# ========================= WINDOWED FFT =========================
num_windows = (N - window_size) // window_interval + 1
print(f"Number of windows: {num_windows}")

plt.figure(figsize=(14, 10))

for i in range(num_windows):
    start = i * window_interval
    end = start + window_size
    segment = y[start:end]
    
    # Apply Hann window
    window = hann(window_size)
    windowed = segment * window
    
    # Zero padding (4x)
    padded = np.pad(windowed, (0, window_size * (padding_factor - 1)), mode='constant')
    
    # FFT
    yf = np.fft.rfft(padded)
    xf = np.fft.rfftfreq(len(padded), d=1.0/sampling_rate)
    
    amplitude = 2.0 / len(padded) * np.abs(yf)
    
    # Limit to desired max frequency
    freq_mask = xf <= max_freq
    xf_plot = xf[freq_mask]
    amp_plot = amplitude[freq_mask]
    
    # Plot each window's spectrum
    plt.subplot(num_windows, 1, i+1)
    plt.plot(xf_plot, amp_plot, color='blue', linewidth=1.2)
    plt.title(f'Window {i+1} (Samples {start}-{end-1})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    if use_log_scale:
        plt.xscale('log')
    
    plt.xlim(0, max_freq)

plt.tight_layout()
plt.suptitle('Frequency Spectra - Individual Windows (4x Zero Padding + Hann Window)', 
             fontsize=14, y=1.02)
plt.show()