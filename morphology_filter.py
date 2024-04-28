import cv2
import matplotlib.pyplot as plt

image = cv2.imread('/home/jinhakim/anaconda3/envs/indCV/pythonProject/son.jpg', cv2.IMREAD_GRAYSCALE)

thres_type = input('Enter thresholding type either "otsu" or "adaptive" : ')

if thres_type == 'otsu':
    # global thresholding(Otsu's method)
    ret, th1 = cv2.threshold(image, -1, 255, cv2.THRESH_OTSU, 0)
    erosion = cv2.morphologyEx(th1, cv2.MORPH_ERODE, (3, 3), iterations=10)
    dilation = cv2.morphologyEx(th1, cv2.MORPH_DILATE, (3, 3), iterations=10)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4, 4)), iterations=5)
    closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4, 4)), iterations=5)

    # title list & image list
    titles = ['Original', 'Global thresholding(Otsu)', 'erosion 10 times', 'dilation 10 times', 'opening 5 times', 'closing 5 times']
    imgs = [image, th1, erosion, dilation, opening, closing]

    for i in range(6):
        plt.subplot(3, 2, i+1), plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
else:
    # adaptive thresholding
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    erosion = cv2.morphologyEx(th2, cv2.MORPH_ERODE, (3, 3), iterations=10)
    dilation = cv2.morphologyEx(th2, cv2.MORPH_DILATE, (3, 3), iterations=10)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)), iterations=5)
    closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)), iterations=5)

    # title list & image list
    titles = ['Original', 'Local thresholding(Adaptive)', 'erosion 10 times', 'dilation 10 times', 'opening 5 times', 'closing 5 times']
    imgs = [image, th2, erosion, dilation, opening, closing]

    for i in range(6):
        plt.subplot(3, 2, i + 1), plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

