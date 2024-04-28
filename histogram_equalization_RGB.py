import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/jinhakim/anaconda3/envs/indCV/pythonProject/balloons.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception('Image file reading error has occurred')

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow('image_rgb', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

channel_type = input('Enter one of channels "R", "G" and "B" : ')

if channel_type == "R":
    # extracting red channel
    channel_red = image_rgb[:, :, 2]
    plt.imshow(channel_red)
    plt.axis('off')
    plt.title('channel_red')
    plt.show()

    # histogram
    hist, bins = np.histogram(channel_red, 256, [0, 256])
    plt.fill(hist)
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.title('histogram_red')
    plt.show()

    # histogram equalization
    red_equalized = cv2.equalizeHist(channel_red)
    hist, bins = np.histogram(red_equalized, 256, [0, 256])
    plt.fill_between(range(256), hist, 0)
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.title('histogram_equalization_red')
    plt.show()

elif channel_type == "G":
    # extracting green channel
    channel_green = image_rgb[:, :, 1]
    plt.imshow(channel_green)
    plt.axis('off')
    plt.title('channel_green')
    plt.show()

    # histogram
    hist, bins = np.histogram(channel_green, 256, [0, 256])
    plt.fill(hist)
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.title('histogram_green')
    plt.show()

    # histogram equalization
    green_equalized = cv2.equalizeHist(channel_green)
    hist, bins = np.histogram(green_equalized, 256, [0, 256])
    plt.fill_between(range(256), hist, 0)
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.title('histogram_equalization_green')
    plt.show()

else:
    # extracting blue channel
    channel_blue = image_rgb[:, :, 0]
    plt.imshow(channel_blue)
    plt.axis('off')
    plt.title('channel_blue')
    plt.show()

    # histogram
    hist, bins = np.histogram(channel_blue, 256, [0, 256])
    plt.fill(hist)
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.title('histogram_blue')
    plt.show()

    # histogram equalization
    blue_equalized = cv2.equalizeHist(channel_blue)
    hist, bins = np.histogram(blue_equalized, 256, [0, 256])
    plt.fill_between(range(256), hist, 0)
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.title('histogram_equalization_blue')
    plt.show()