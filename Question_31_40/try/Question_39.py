# JPEG圧縮のStep3 YCbCr表色系
# RGB系からY, Cb, Cr系へ変換を行いコントラストを暗くする
#変換式
"""
Y = 0.299 * R + 0.5870 * G + 0.114 * B
Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
--------------------------------------------------
R = Y + (Cr - 128) * 1.402
G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
B = Y + (Cb - 128) * 1.7718
"""
import cv2
import numpy as np
import time

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.result_img = None
        self.YCbCr_img = None
    
    def BGR2YCbCr(self):
        H, W, C = self.img.shape
        b = self.img[:,:, 0]
        g = self.img[:,:, 1]
        r = self.img[:,:, 2]
        out = np.zeros((H, W, C), dtype=np.float32)

        out[..., 0] = 0.299 * r + 0.5870 * g + 0.114 * b
        out[..., 1] = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
        out[..., 2] = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

        self.YCbCr_img = out.copy()

        #今回はここで変換までする
        out[..., 0] *= 0.7
        self.result_img = out.copy()
    
    def YCbCr2BGR(self):
        H, W, C = self.img.shape
        out = np.zeros((H, W, C), dtype=np.float32)
        Y = self.result_img[:,:, 0]
        Cb = self.result_img[:,:, 1]
        Cr = self.result_img[:,:, 2]
        out[:,:, 2] = Y + (Cr - 128) * 1.402
        out[:,:, 1] = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
        out[:,:, 0] = Y + (Cb - 128) * 1.7718

        self.result_img = out.copy().astype(np.uint8)
    

img_file_path = "../imori.jpg"
process = process_class(img_file_path)
process.BGR2YCbCr()
process.YCbCr2BGR()
cv2.imwrite("./result/Question_39.jpg", process.result_img)
    
