

import tensorflow as tf
import cv2
import cupy
import  numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img = cv2.imread('D:\\ML_learning\\ESRGAN-master\\LR\\1.png')
    print(img)
    resize_0 = cv2.resize(img,(800,800),interpolation=cv2.INTER_NEAREST)
    resize_1 = cv2.resize(img, (800,800), interpolation=cv2.INTER_LINEAR)
    resize_2 = cv2.resize(img, (800,800), interpolation=cv2.INTER_AREA)
    resize_3 = cv2.resize(img, (800,800), interpolation=cv2.INTER_CUBIC)
    resize_4 = cv2.resize(img, (800,800), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('\\0.png',resize_0)
    cv2.imwrite('\\1.png', resize_1)
    cv2.imwrite('\\2.png', resize_2)
    cv2.imwrite('\\3.png', resize_3)
    cv2.imwrite('\\4.png', resize_4)

    plt.show()




