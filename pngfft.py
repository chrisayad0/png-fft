from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ========================= CONFIG =========================
image_path = r"chart.png"   # <<< CHANGE THIS
sampling_rate = 40000                         # <<< Adjust if needed (for 2000 samples → 4000 Hz sampling rate)
max_freq = 250                              # Set to None to show full Nyquist, or set a number (e.g. 2000)
# ========================================================

# Load image
img = Image.open(image_path).convert('RGB')
img_array = np.array(img)

print(f"Image loaded: {img_array.shape}")

# Find red waveform
r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
mask = (r > 180) & (g < 100) & (b < 100)

height, width = mask.shape[:2]
y_values = np.zeros(width)

for x in range(width):
    column = mask[:, x]
    if np.any(column):
        y_coords = np.where(column)[0]
        y_values[x] = np.mean(y_coords)
    else:
        y_values[x] = np.nan

# Interpolate missing values
y_values = np.interp(np.arange(width), 
                     np.where(~np.isnan(y_values))[0], 
                     y_values[~np.isnan(y_values)])

# Flip Y axis
y = height - y_values

# Normalize
y = y - np.mean(y)
y = y / np.max(np.abs(y))

# ========================= FFT =========================
N = len(y)
yf = np.fft.rfft(y)
xf = np.fft.rfftfreq(N, d=1.0/sampling_rate)

amplitude = 2.0/N * np.abs(yf)

# Limit frequency range
if max_freq is None:
    max_freq = sampling_rate / 2   # Nyquist frequency

# Find indices up to max_freq
freq_mask = xf <= max_freq
xf_plot = xf[freq_mask]
amp_plot = amplitude[freq_mask]

# ========================= PLOTTING =========================
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Time domain
axs[0].plot(y, color='red', linewidth=1.5)
axs[0].set_title('Time Domain Waveform')
axs[0].set_xlabel('Sample')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Frequency domain (limited)
axs[1].plot(xf_plot, amp_plot, color='blue', linewidth=1.5)
axs[1].set_title('Frequency Spectrum (FFT)')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

# Set x limit clearly
axs[1].set_xlim(0, max_freq)

plt.tight_layout()
plt.show()

# Optional: Save plot
# plt.savefig('frequency_plot_limited.png', dpi=300)