# JPEG圧縮のStep2
# 離散コサイン変換(DCT) + 量子化
# 量子化とは値をあらかじめ決定された区分ごとに値を大まかに丸め込む作業のこと
# 量子化の代表例 : floor, ceil, roundなどが似た計算
# JPEG画像ではDCT係数を量子化テーブルにのっとって量子化する
import cv2
import numpy as np
import time 

class process_class:
    def __init__(self, img_file_path, T, K, channel):
        self.img = cv2.imread(img_file_path)
        self.F = None
        self.idct_img = None
        self.T = T
        self.K = K
        self.channel = channel
    
    def w(self, x, y, u, v):
        cu = 1.
        cv = 1.
        if u == 0:
            cu /= np.sqrt(2)
        if v == 0:
            cv /= np.sqrt(2)
        theta = np.pi / (2*self.T)
        
        return ((2* cu * cv / T))*np.cos((2*x+1)*u*theta)*np.cos((2*y+1)*v*theta)
    
    def dct(self):
        H, W, _ = self.img.shape
        out = np.zeros((H, W, self.channel), dtype=np.float32)

        for c in range(self.channel):
            for yi in range(0, H, self.T):
                for xi in range(0, W, self.T):
                    for u in range(self.T):
                        for v in range(self.T):
                            for y in range(self.T):
                                for x in range(self.T):
                                    out[yi+u, xi+v, c] += self.img[yi+y, xi+x, c] * self.w(x,y,u,v)

        self.F = out

    def idct(self):
        H, W, _ = self.img.shape
        out = np.zeros((H, W, self.channel), dtype=np.float32)

        for c in range(self.channel):
            for yi in range(0, H, self.T):
                for xi in range(0, W, self.T):
                    for y in range(self.T):
                        for x in range(self.T):
                            for u in range(self.K):
                                for v in range(self.K):
                                    out[yi+y, xi+x, c] += self.F[yi+u, xi+v, c]*self.w(x,y,u,v)
        
        out = np.clip(out, 0, 255)
        out = np.round(out).astype(np.uint8)

        self.idct_img = out
    
    def quantization(self):
        H, W, _ = self.F.shape
        temp = self.F.copy()

        Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                    (12, 12, 14, 19, 26, 58, 60, 55),
                    (14, 13, 16, 24, 40, 57, 69, 56),
                    (14, 17, 22, 29, 51, 87, 80, 62),
                    (18, 22, 37, 56, 68, 109, 103, 77),
                    (24, 35, 55, 64, 81, 104, 113, 92),
                    (49, 64, 78, 87, 103, 121, 120, 101),
                    (72, 92, 95, 98, 112, 100, 103, 99),) , dtype=np.float32)
        
        for ys in range(0, H, self.T):
            for xs in range(0, W, self.T):
                for c in range(self.channel):
                    temp[ys:ys+self.T, xs: xs+self.T, c] = np.round(self.F[ys:ys+self.T, xs:xs+self.T, c]/Q)*Q

        self.F = temp
    
    def MSE(self, img1, img2):
        H, W, _ = img1.shape
        mse = np.sum((img1 - img2)**2) / (H*W*self.channel)
        return mse
    
    def PSNR(self, vmax=255):
        mse = self.MSE(self.img, self.idct_img)
        return 10*np.log10(vmax * vmax / mse)
    
    def BITRATE(self):
        return 1. * self.T * self.K * self.K / self.T/ self.T

start_time = time.time()
img_file_path = "../imori.jpg"
T = 8
K = 8
channel = 3
process = process_class(img_file_path, T, K, channel)
process.dct()
orig_F = process.F.copy()
process.quantization()
after_F = process.F
process.idct()
mse = process.MSE(process.img, process.idct_img)
psnr = process.PSNR()
bitrate = process.BITRATE()
print(f"mse : {mse}")
print(f"psnr : {psnr}")
print(f"bitrate : {bitrate}")
# パワースペクトルを見るための処理
# 絶対値 + 対数 + 正規化 + クリップ + 型変換
orig_F = np.abs(orig_F)
after_F = np.abs(after_F)
orig_F = np.log(1+orig_F)
after_F = np.log(1+after_F)
orig_F = cv2.normalize(orig_F, None, 0, 255, cv2.NORM_MINMAX)
after_F = cv2.normalize(after_F, None, 0, 255, cv2.NORM_MINMAX)
orig_F = np.clip(orig_F, 0, 255).astype(np.uint8)
after_F = np.clip(after_F, 0, 255).astype(np.uint8)

cv2.imwrite("./result/Question_38.jpg", process.idct_img)
cv2.imwrite("./result/Question_38_origF.jpg", orig_F)
cv2.imwrite("./result/Question_38_afterF.jpg", after_F)
end_time = time.time()
print(f"処理時間 : {end_time - start_time:.4f} sec")