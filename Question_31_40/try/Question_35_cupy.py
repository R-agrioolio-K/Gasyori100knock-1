# DFTを行いバンドパスフィルタを適用するプログラム
# バンドパスフィルタは低周波成分と高周波成分の中間の周波数成分のみを通す
import cv2
import numpy as np
import cupy as cp
import time

class process_class:
    def __init__(self, img_file_path, L = 128, K=128, channel=3):
        self.cpu_img = cv2.imread(img_file_path)
        self.gpu_img = cp.array(self.cpu_img)
        self.G = None
        self.L = L
        self.K = K
        self.channel = 3
        self.idft_img = None
    
    def dft(self):
        H, W, _ = self.gpu_img.shape
        G = cp.zeros((self.L, self.K, self.channel), dtype=cp.complex128)

        y = cp.arange(H).repeat(W).reshape(H, -1)
        x = cp.tile(cp.arange(W), (H, 1))

        for c in range(self.channel):
            for l in range(self.L):
                for k in range(self.K):
                    G[l, k, c] = (cp.sum(self.gpu_img[..., c] 
                    * cp.exp(-2j * cp.pi * (x*k/self.K + y*l/self.L))))/cp.sqrt(self.K*self.L)
        
        self.G = G
    
    def idft(self):
        H, W, _ = self.G.shape
        out = cp.zeros((H, W, self.channel), dtype=cp.float32)

        # 画像を出力するため画像の配列番号を格納した配列を作成する
        x = cp.tile(cp.arange(W), (H, 1))
        y = cp.arange(H).repeat(W).reshape(H, -1)

        for c in range(self.channel):
            for l in range(H):
                for k in range(W):
                    out[l, k, c] = (cp.abs(cp.sum(self.G[..., c]
                    * cp.exp(2j * cp.pi * (x*k/W + y*l/H))))/cp.sqrt(W*H))
        
        out = cp.clip(out, 0, 255)
        out = out.astype(np.uint8).get()

        self.idft_img = out
    
    def bpf(self, ratio1=0.1, ratio2=0.5):
        H, W, _ = self.G.shape
        G = self.G

        #配列の位置を変更する
        _G = cp.zeros_like(G)
        _G[:H//2, :W//2] = G[H//2:, W//2:]
        _G[:H//2, W//2:] = G[H//2:, :W//2]
        _G[H//2:, :W//2] = G[:H//2, W//2:]
        _G[H//2:, W//2:] = G[:H//2, :W//2]

        #中央からの距離を算出する
        x = cp.tile(cp.arange(W), (H, 1))
        y = cp.arange(H).repeat(W).reshape(H, -1)
        _x = x - W//2
        _y = y - H//2
        r = cp.sqrt(_x**2 + _y**2)
        mask = cp.ones((H, W), dtype=cp.float32)
        mask[(r < (W//2 *ratio1)) | ((r > (W//2*ratio2)))] = 0

        mask = np.repeat(mask, self.channel).reshape(H, W, self.channel)

        _G *= mask

        # 元の位置に戻す
        G[:H//2, :W//2] = _G[H//2:, W//2:]
        G[:H//2, W//2:] = _G[H//2:, :W//2]
        G[H//2:, :W//2] = _G[:H//2, W//2:]
        G[H//2:, W//2:] = _G[:H//2, :W//2]

        self.G = G

start_time = time.time()
img_file_path = "../imori.jpg"
process = process_class(img_file_path)
process.dft()
G_orig = process.G
process.bpf()
process.idft()
G_after = process.G

# パワースペクトルを表示するための処理を行う
# 絶対値 -> 対数 -> 正規化 -> クリップ -> RAMに戻す
G_orig = cp.abs(G_orig)
G_after = cp.abs(G_after)
G_orig = cp.log(1+G_orig)
G_after = cp.log(1+G_after)
G_after = G_after.get()
a = 0
b = 255
c = G_orig.min()
d = G_orig.max()
G_orig = (b - a)/(d - c) * (G_orig - c) + a
G_after = cv2.normalize(G_after, None, 0, 255, cv2.NORM_MINMAX)
G_orig = cp.clip(G_orig, 0, 255).astype(cp.uint8).get()
G_after = np.clip(G_after, 0, 255).astype(np.uint8)

cv2.imwrite("./result/Question_35.jpg", process.idft_img)
cv2.imwrite("./result/Question_35_ps_orig.jpg", G_orig)
cv2.imwrite("./result/Question_35_ps_after.jpg", G_after)

end_time = time.time()
print(f"実行時間 : {end_time - start_time:.4f} sec")



        
