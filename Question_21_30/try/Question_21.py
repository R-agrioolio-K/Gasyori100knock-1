# ヒストグラム正規化をするためのプログラム
import cv2
import numpy as np
import matplotlib.pyplot as plt

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.normalization_img = None
    
    def hist_normalization(self, a=0, b=255):
        #画像に含まれる最大値と最小値を取得
        c = self.img.min()
        d = self.img.max()

        out = self.img.copy()

        #正規化の実施
        out = (b - a) / (d - c) * (out - c) + a
        out[out < a] = a
        out[out > b] = b
        out = out.astype(np.uint8)

        self.normalization_img = out

img_file_path = "../imori_dark.jpg"
process = process_class(img_file_path)

process.hist_normalization()

#画像の保存
cv2.imwrite("./result/Question_21.jpg", process.normalization_img)

# 正規化ヒストグラムの保存
plt.hist(process.normalization_img.ravel(), bins=255, rwidth=0.8, range=(0,255))
plt.savefig("./result/Question_21_hist.jpg")
plt.close()

# 処理前ヒストグラムの保存
plt.hist(process.img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("./result/hist.jpg")
plt.close()
