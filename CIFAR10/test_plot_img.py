import numpy as np
import matplotlib.pyplot as plt
import cv2

img=np.load('mask.npy')
desired_size = (16, 16)
resized_image = cv2.resize(img, desired_size)

transposed_arr = resized_image.transpose(2, 0, 1)
