# 最近傍補完法を実行するプログラム
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.rescale = None
    
    def Nearest_Neighbor(self, ax = 1, ay = 1):
        H, W, C = self.img.shape
        print(f"元画像 : H = {H}, W = {W}, C = {C}")

        aH = int(ay * H)
        aW = int(ax * W)

        y = np.arange(aH).repeat(aW).reshape(aH, -1)
        x = np.tile(np.arange(aW), (aH, 1))
        y = np.round(y / ay).astype(np.int8)    #縮尺・拡大処理による原画像の参照座標を算出
        x = np.round(x / ax).astype(np.int8)    #縮尺・拡大処理により原画像の参照座標を算出

        out = self.img[y, x].copy() # 2次元配列を使うことでそれぞれの行列の要素を抜き出すことができる

        out = out.astype(np.uint8)

        self.rescale = out
        rH, rW, rC = out.shape
        print(f"返還後 : H = {rH}, W = {rW}, C = {rC}")

img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.Nearest_Neighbor(ax=0.8, ay=0.5)

cv2.imwrite("./result/Question_25.jpg", process.rescale)
