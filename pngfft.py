from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ========================= CONFIG =========================
image_path = r"chart.png"   # <<< CHANGE THIS TO YOUR FILE PATH
sampling_rate = 44100                        # Change if you know the real sampling rate (Hz)
# ========================================================

# Load image
img = Image.open(image_path).convert('RGB')
img_array = np.array(img)

print(f"Image loaded: {img_array.shape}")

# Find the red waveform (bright red pixels)
# The waveform in your image is bright red (255, ~0, ~0)
r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

# Mask for red line (adjust threshold if needed)
mask = (r > 180) & (g < 100) & (b < 100)

# Get y-coordinates of the waveform for each x
height, width = mask.shape[:2]
y_values = np.zeros(width)

for x in range(width):
    column = mask[:, x]
    if np.any(column):
        # Take the middle of the red pixels in this column
        y_coords = np.where(column)[0]
        y_values[x] = np.mean(y_coords)   # or y_coords[0] for top edge
    else:
        y_values[x] = np.nan

# Fill missing values with linear interpolation
y_values = np.interp(np.arange(width), 
                     np.where(~np.isnan(y_values))[0], 
                     y_values[~np.isnan(y_values)])

# Flip y-axis because image coordinates are top-down
y = height - y_values

# Normalize signal
y = y - np.mean(y)
y = y / np.max(np.abs(y))

# ========================= FFT =========================
N = len(y)
yf = np.fft.rfft(y)
xf = np.fft.rfftfreq(N, d=1.0/sampling_rate)

# Amplitude spectrum
amplitude = 2.0/N * np.abs(yf)

# Plot both Time Domain and Frequency Domain
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Time domain
axs[0].plot(y, color='red', linewidth=1.5)
axs[0].set_title('Time Domain Waveform')
axs[0].set_xlabel('Sample')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Frequency domain
axs[1].plot(xf, amplitude, color='blue', linewidth=1.5)
axs[1].set_title('Frequency Spectrum (FFT)')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

# Optional: zoom into lower frequencies
# axs[1].set_xlim(0, 5000)   # Uncomment and adjust to zoom

plt.tight_layout()
plt.show()

# Optional: Save the frequency plot
# plt.savefig('frequency_plot.png', dpi=300)