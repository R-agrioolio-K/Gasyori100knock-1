# 離散コサイン変換による画像の劣化を求める
# PSNR(Peak Signal to Noise Ratio)、信号対雑音比を求め劣化を定量的に評価する
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class process_class:
    def __init__(self, img_file_path, T, K, channel):
        self.img = cv2.imread(img_file_path).astype(np.float32)
        self.T = T
        self.K = K
        self.channel = channel
        self.F = None
        self.idct_img = None
    
    def w(self, x, y, u, v):
        cu = 1.
        cv = 1.
        if u == 0:
            cu = 1/np.sqrt(2)
        if v == 0:
            cv /= np.sqrt(2)
        theta = np.pi / (2*self.T)
        return ((2*cu*cv / self.T)*np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

    def dct(self):
        H, W, _ = self.img.shape
        F = np.zeros((H, W, self.channel), dtype=np.float32)

        for c in range(self.channel):
            for yi in range(0, H, T):
                for xi in range(0, W, T):
                    for v in range(T):
                        for u in range(T):
                            for y in range(T):
                                for x in range(T):
                                    F[v+yi, u+xi, c] += self.img[y+yi, x+xi, c] * self.w(x, y, u, v)
        
        self.F = F
    
    def idct(self):
        H, W, _ = self.F.shape

        out = np.zeros((H, W, channel), dtype=np.float32)

        for c in range(self.channel):
            for yi in range(0, H, T):
                for xi in range(0, W, T):
                    for y in range(T):
                        for x in range(T):
                            for v in range(K):
                                for u in range(K):
                                    out[y+yi, x+xi, c] += self.F[v+yi, u+xi, c] * self.w(x,y,u,v)
        
        out = np.clip(out, 0, 255)
        out = np.round(out).astype(np.uint8)

        self.idct_img = out
    
    def MSE(self):
        H, W, _ = self.img.shape
        mse = np.sum((self.img - self.idct_img) ** 2) / (H*W*self.channel)
        return mse
    
    def PSNR(self, vmax=255):
        return 10*np.log10(vmax*vmax / mse)

    def BITRATE(self):
        return 1. * self.T * self.K * self.K / self.T / self.T


start_time = time.time()
img_file_path = "../imori.jpg"
T = 8
K = 4
channel = 3
process = process_class(img_file_path, T, K, channel)
process.dct()
process.idct()
mse = process.MSE()
psnr = process.PSNR()
bitrate = process.BITRATE()

print("MSE : ", mse)
print("PSNR : ", psnr)
print("bitrate : ",bitrate)

F = process.F
# 変換数値Fを見えるようにする
# 絶対値 + 対数 + 正規化 + クリップ
F = abs(F)
F = np.log(1+F)
a = 0
b = 255
c = F.min()
d = F.max()
F = (b - a)/(d - c) * (F - c) + a
F = np.clip(F, 0, 255).astype(np.uint8)
cv2.imwrite("./result/Question_37.jpg", process.idct_img)
cv2.imwrite("./result/Question_37_F.jpg", F)
end_time = time.time()
print(f"処理時間 : {end_time - start_time:.4f} sec")



