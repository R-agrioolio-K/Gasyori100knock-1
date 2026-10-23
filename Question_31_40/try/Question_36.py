# 離散コサイン変換(DCT : Discrete Cosine Transformation)を行うプログラムファイル
# JPEG圧縮の一部として扱われている?
# 今回は深いネストを使っているためnumpyのほうが実行が速い
import cv2
import numpy as np
import cupy as cp
import time

class process_class:
    def __init__(self, img_file_path, T, K, channel):
        self.cpu_img = cv2.imread(img_file_path)
        self.T = T
        self.K = K
        self.channel = channel
        self.F = None
        self.idct_img = None
    
    def w(self, x, y, u, v):
        cu = 1
        cv = 1
        if u == 0:
            cu /= np.sqrt(2)
        if v == 0:
            cv /= np.sqrt(2)
        theta = np.pi / (2 * self.T)

        return ((2 * cu * cv / self.T) * np.cos((2*x+1)*u*theta) * np.cos((2*y + 1)*v*theta))
    
    # 離散コサイン変換(DCT)を行う関数
    def dct(self):
        H, W, _ = self.cpu_img.shape
        F = np.zeros((H, W, self.channel), dtype=np.float32)

        for c in range(self.channel):
            for yi in range(0, H, self.T):
                for xi in range(0, W, self.T):
                    for v in range(self.T):
                        for u in range(self.T):
                            for y in range(self.T):
                                for x in range(self.T):
                                    F[v+yi, u+xi, c] += self.cpu_img[y+yi, x+xi, c] * self.w(x,y,u,v)
        
        self.F = F

    # 逆離散コサイン変換(IDCT)を行う関数
    def idct(self):
        H, W, _ = self.F.shape
        out = np.zeros((H, W, self.channel), dtype=np.float32)

        for c in range(self.channel):
            for yi in range(0, H, self.T):
                for xi in range(0, W, self.T):
                    for y in range(self.T):
                        for x in range(self.T):
                            for v in range(self.K):
                                for u in range(self.K):
                                    out[y+yi, x+xi, c] += self.F[v+yi, u+xi, c] * self.w(x,y,u,v)
        
        out = np.clip(out, 0, 255)
        out = np.round(out).astype(np.uint8)
        self.idct_img = out

start_time = time.time()
img_file_path = "../imori.jpg"
process = process_class(img_file_path, T=8, K=8, channel=3)
process.dct()
F = process.F
process.idct()
# Fを見やすくするために
# 絶対値 + 対数 + 正規化 + クリップ
F = np.abs(F)
F = np.log(1+F)
a = 0
b = 255
c = F.min()
d = F.max()
F = (b - a)/(d - c) * (F - c) + a
F = np.clip(F, 0, 255).astype(np.uint8)
cv2.imwrite("./result/Question_36.jpg", process.idct_img)
cv2.imwrite("./result/Question_36_F.jpg", F)
end_time = time.time()
print(f"実行時間 : {end_time - start_time:4f} sec")