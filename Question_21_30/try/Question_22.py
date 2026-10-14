# ヒストグラム操作を行うプログラム
# ここではヒストグラムの平均値を128, 標準偏差を52になるように操作する
# これはヒストグラムのダイナミックレンジを変更するのではなくヒストグラムを平坦化する
import cv2
import numpy as np
import matplotlib.pyplot as plt

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.result = None
    
    def hist_alter(self, m0 = 128, s0 = 52):
        m = np.mean(self.img)
        s = np.std(self.img)

        out = self.img.copy()

        # 変換処理
        out = ((s0/s) * (out - m)) + m0
        out[out < 0] = 0
        out[out > 255] = 255

        self.result = out.astype(np.uint8)

img_file_path = "../imori_dark.jpg"
process = process_class(img_file_path)
process.hist_alter()

cv2.imwrite("./result/Question_22.jpg",process.result)

plt.hist(process.result.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("./result/Question_22_hist.jpg")
plt.close()