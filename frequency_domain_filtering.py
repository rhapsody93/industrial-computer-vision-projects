import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/jinhakim/anaconda3/envs/indCV/pythonProject/son.jpg',
                   cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

# Discrete Fourier Transform
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)

# move to frequency domain
shifted = np.fft.fftshift(fft, axes=[0, 1])
magnitude = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
magnitude = np.log(magnitude)

# visualizing frequency domain shifted image
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.title('frequency domain')


rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2

# typing random integer for circle_01 radius
circle_01 = int(input('Enter the radius of circle_01 : '))
mask_circle_01 = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask_circle_01, (center_col, center_row), circle_01, (1, 1), -1)

# typing random integer for circle_02 radius
circle_02 = int(input('Enter the radius of circle_02 : '))
mask_circle_02 = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask_circle_02, (center_col, center_row), circle_02, (1, 1), -1)

# bandpass filter creation (difference between mask_circle_01 and mask_circle_02
bandpass_mask = mask_circle_02 - mask_circle_01

# applying bandpass filter into frequency domain
filtered_shifted = shifted * bandpass_mask

# inverse transform
filtered_fft = np.fft.ifftshift(filtered_shifted, axes=[0, 1])
restored = cv2.idft(filtered_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

# visualizing result
plt.subplot(122)
plt.axis('off')
plt.imshow(restored, cmap='gray')
plt.title('filtered image')

plt.tight_layout()
plt.show()

