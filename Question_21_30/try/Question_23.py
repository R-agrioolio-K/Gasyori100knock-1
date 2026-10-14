# ヒストグラム平坦化を行うプログラム
# ヒストグラム平坦化はヒストグラム値を近郊にする走査である
import cv2
import numpy as np
import matplotlib.pyplot as plt

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.result = None
    
    # ヒストグラム平坦化(histgram equalization)
    def hist_equal(self, z_max=255):
        H, W, C = self.img.shape
        S = H * W * C * 1.

        out = self.img.copy()

        sum_h = 0.

        for i in range(1, 255):
            ind = np.where(self.img == i)    # 画素値がiと同じもののみ取り出す
            sum_h += len(self.img[ind])
            z_prime = ((z_max / S) * sum_h)
            out[ind] = z_prime
        
        out = out.astype(np.uint8)

        self.result = out

img_file_path = "../imori_dark.jpg"
process = process_class(img_file_path)

process.hist_equal()

cv2.imwrite("./result/Question_23.jpg", process.result)

plt.hist(process.result.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("./result/Question_23_hist.jpg")
plt.close()

