import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/jinhakim/anaconda3/envs/indCV/pythonProject/Lenna.png', cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# adding noise
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 255).astype(np.uint8)
noised_rgb = cv2.cvtColor(noised, cv2.COLOR_BGR2RGB)

# applying gaussian blur
gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
gauss_blur_rgb = cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB)

# applying median blur
median_blur = cv2.medianBlur(noised, 7)
median_blur_rgb = cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB)

# applying bilateral filter
bilateral_filter = cv2.bilateralFilter(noised, -1, 0.3, 10)
bilateral_filter_rgb = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB)

# absolute value (original image and filter applied image)
abs_gauss_blur = np.abs(image_rgb.astype(np.float32) - gauss_blur_rgb.astype(np.float32)).astype(np.uint8)
abs_median_blur = np.abs(image_rgb.astype(np.float32) - median_blur_rgb.astype(np.float32)).astype(np.uint8)
abs_bilateral_filter = np.abs(image_rgb.astype(np.float32) - bilateral_filter_rgb.astype(np.float32)).astype(np.uint8)

# visualizing result
titles = ['original', 'noised', 'gaussian blur', 'abs gaussian vlue',
          'median blur', 'abs median blur', 'bilateral filter', 'abs bilateral filter']
images = [image_rgb, noised_rgb, gauss_blur_rgb, abs_gauss_blur,
          median_blur_rgb, abs_median_blur, bilateral_filter_rgb, abs_bilateral_filter]

plt.figure(figsize=(10, 10))

for i in range(8):
    plt.subplot(4, 2, i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
