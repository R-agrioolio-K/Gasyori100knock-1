# JPEG圧縮(Step3)YCbCr + DCT + 量子化
# YCbCr表色系に変換しDCT後YをQ1でCb,CrをQ2で量子化しIDCTで復元する
# なお画像の容量を比較する
"""
-----------BGR2YCbCr変換-----------------
Y = 0.299 * R + 0.5870 * G + 0.114 * B
Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
------------YCbCr2BGR変換-----------------
R = Y + (Cr - 128) * 1.402
G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
B = Y + (Cb - 128) * 1.7718

----------------------量子化テーブル---------------------------------
Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
               (12, 12, 14, 19, 26, 58, 60, 55),
               (14, 13, 16, 24, 40, 57, 69, 56),
               (14, 17, 22, 29, 51, 87, 80, 62),
               (18, 22, 37, 56, 68, 109, 103, 77),
               (24, 35, 55, 64, 81, 104, 113, 92),
               (49, 64, 78, 87, 103, 121, 120, 101),
               (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
               (18, 21, 26, 66, 99, 99, 99, 99),
               (24, 26, 56, 99, 99, 99, 99, 99),
               (47, 66, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)

"""
import cv2
import numpy as np
import time

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.result_img = None
        self.before_F = None
        self.after_F = None
        self.tmp = self.img.astype(np.float32)
    
    def trans(self):
        self.result_img = np.clip(self.tmp, 0, 255).astype(np.uint8)

    def BGR2YCbCr(self):
        img = self.tmp
        H, W, C = img.shape
        B = img[:,:, 0]
        G = img[:,:, 1]
        R = img[:,:, 2]
        out = np.zeros((H, W, C), dtype=np.float32)
        
        out[..., 0] = 0.299 * R + 0.5870 * G + 0.114 * B
        out[..., 1] = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
        out[..., 2] = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
        self.tmp = out
        
    
    def YCbCr2BGR(self):
        img = self.tmp
        H, W, C = img.shape
        Y = img[:,:, 0]
        Cb = img[:,:, 1]
        Cr = img[:,:, 2]
        out = np.zeros((H, W, C), dtype=np.float32)
        out[..., 2] = Y + (Cr - 128) * 1.402
        out[..., 1] = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
        out[..., 0] = Y + (Cb - 128) * 1.7718

        self.tmp = out
    
    def dct_w(self, x, y, u, v, T):
        cu = 1.
        cv = 1.
        if u == 0:
            cu /= np.sqrt(2)
        if v == 0:
            cv /= np.sqrt(2)
        theta = np.pi/(2*T)
        return 2*cu*cv*np.cos((2*x+1)*u*theta)*np.cos((2*y+1)*v*theta)/T

    def dct(self, T=8, K=8, channel=3):
        H, W, _ = self.tmp.shape
        img = self.tmp
        out = np.zeros((H, W, channel), dtype=np.float32)

        for c in range(channel):
            for iy in range(0, H, T):
                for ix in range(0, W, T):
                    for v in range(T):
                        for u in range(T):
                            for x in range(T):
                                for y in range(T):
                                    out[iy+v, ix+u, c] += img[iy+y, ix+x, c]*self.dct_w(x,y,u,v,T)
        
        self.tmp = out
    
    def idct(self, T=8, K=8, channel=3):
        H, W, _ = self.tmp.shape
        img = self.tmp
        out = np.zeros((H, W, channel), dtype=np.float32)

        for c in range(channel):
            for iy in range(0, H, T):
                for ix in range(0, W, T):
                    for x in range(T):
                        for y in range(T):
                            for v in range(K):
                                for u in range(K):
                                    out[iy+y, ix+x, c] += img[iy+v, ix+u, c]*self.dct_w(x,y,u,v,T)
        
        self.tmp = out
    
    def quantum(self, T=8, K=8, channel=3):
        F = self.tmp
        H, W, _ = self.tmp.shape
        Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
               (12, 12, 14, 19, 26, 58, 60, 55),
               (14, 13, 16, 24, 40, 57, 69, 56),
               (14, 17, 22, 29, 51, 87, 80, 62),
               (18, 22, 37, 56, 68, 109, 103, 77),
               (24, 35, 55, 64, 81, 104, 113, 92),
               (49, 64, 78, 87, 103, 121, 120, 101),
               (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

        Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
                    (18, 21, 26, 66, 99, 99, 99, 99),
                    (24, 26, 56, 99, 99, 99, 99, 99),
                    (47, 66, 99, 99, 99, 99, 99, 99),
                    (99, 99, 99, 99, 99, 99, 99, 99),
                    (99, 99, 99, 99, 99, 99, 99, 99),
                    (99, 99, 99, 99, 99, 99, 99, 99),
                    (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)
        
        for ys in range(0, H, T):
            for xs in range(0, W, T):
                F[ys:ys+T, xs:xs+T, 0] = np.round(F[ys:ys+T, xs:xs+T, 0]/Q1)*Q1
                F[ys:ys+T, xs:xs+T, 1] = np.round(F[ys:ys+T, xs:xs+T, 1]/Q2)*Q2
                F[ys:ys+T, xs:xs+T, 2] = np.round(F[ys:ys+T, xs:xs+T, 2]/Q2)*Q2

        self.tmp = F

    def JPEG(self):
        self.BGR2YCbCr()
        self.dct()
        self.before_F = self.tmp.copy()
        self.quantum()
        self.after_F = self.tmp.copy()
        self.idct()
        self.YCbCr2BGR()
        self.trans()
        
    def MSE(self, img1, img2):
        H, W, c = img1.shape
        mse = np.sum((img1 - img2)**2)/(H*W*c)
        return mse
        
    def PSNR(self, mse, vmax=255):
        return 10*np.log10(vmax*vmax / mse)
        
    def BITRATE(self, T, K):
        return 1.* T * K * K / T / T

img_file_path = "../imori.jpg"
T = 8
K = 8
process = process_class(img_file_path)
process.JPEG()
mse = process.MSE(process.img, process.result_img)
psnr = process.PSNR(mse)
bitrate = process.BITRATE(T, K)
before_F = process.before_F
after_F = process.after_F
# パワースペクトルを見るための処理
# 絶対値 + 対数 + 正規化 + クリップ + 型変換
before_F = np.abs(before_F)
after_F = np.abs(after_F)
before_F = np.log(1+before_F)
after_F = np.log(1+after_F)
before_F = cv2.normalize(before_F, None, 0, 255, cv2.NORM_MINMAX)
after_F = cv2.normalize(after_F, None, 0, 255, cv2.NORM_MINMAX)
before_F = np.clip(before_F, 0, 255).astype(np.uint8)
after_F = np.clip(after_F, 0, 255).astype(np.uint8)
print("mse : ", mse)
print("psnr : ", psnr)
print("bitrate : ", bitrate)
cv2.imwrite("./result/Question_40.jpg", process.result_img)
cv2.imwrite("./result/Question_40_before_F.jpg", before_F)
cv2.imwrite("./result/Question_40_after_F.jpg", after_F)

        

