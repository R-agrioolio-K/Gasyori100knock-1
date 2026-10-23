import cv2
import numpy as np
import cupy as cp  # 1. CuPyのインポート
import time

class process_class:
    def __init__(self, img_file_path, K = 128, L = 128, channel = 3):
        # 画像読み込みはOpenCVなので一旦CPU(NumPy)で読み込み、GPU(CuPy)に転送する
        img_cpu = cv2.imread(img_file_path)
        self.img = cp.array(img_cpu)  # CPU -> GPUへ転送
        
        self.G = None
        self.idft_img = None
        self.K = K
        self.L = L
        self.channel = channel
    
    def dft(self):
        H, W, _ = self.img.shape

        # np -> cp に変更
        y = cp.arange(H).repeat(W).reshape(H, -1)
        x = cp.tile(cp.arange(W), (H, 1)) 
        out = cp.zeros((self.L, self.K, self.channel), dtype=cp.complex128)

        # dftによる変換（GPUだとこれでも動きますが、本来はcp.fftを使うと一瞬で終わります）
        for c in range(self.channel):
            for l in range(self.L):
                for k in range(self.K):
                    out[l, k, c] = (cp.sum(self.img[..., c] 
                    * cp.exp(-2j * cp.pi *((k*x/self.K) + (l*y/self.L))))/cp.sqrt(H*W))
        
        self.G = out
    
    def idft(self):
        H, W, _ = self.G.shape
        out = cp.zeros((H, W, self.channel), dtype=cp.float32)

        y = cp.arange(H).repeat(W).reshape(H, -1)
        x = cp.tile(cp.arange(W), (H, 1))
        
        for c in range(self.channel):
            for l in range(self.L):
                for k in range(self.K):
                    out[l, k, c] = (cp.abs(cp.sum(self.G[...,c] 
                    * cp.exp(2j * cp.pi * (x*k/W + y*l/H)))) / cp.sqrt(W*H))
        
        # クリッピング
        out = cp.clip(out, 0, 255)
        out = out.astype(cp.uint8)

        # OpenCVで扱えるように CPU(NumPy) に戻す (.get())
        self.idft_img = out.get()
    
    def hpf(self, ratio = 0.1):
        H, W, _ = self.G.shape
        G = self.G

        # 位置を移動する
        _G = cp.zeros_like(G)
        _G[:H//2, :W//2] = G[H//2:, W//2:]
        _G[:H//2, W//2:] = G[H//2:, :W//2]
        _G[H//2:, :W//2] = G[:H//2, W//2:]
        _G[H//2:, W//2:] = G[:H//2, :W//2]

        # マスクを作成する
        x = cp.tile(cp.arange(W), (H, 1))
        y = cp.arange(H).repeat(W).reshape(H, -1)
        _x = x - W // 2
        _y = y - H // 2

        r = cp.sqrt(_x**2 + _y**2)
        mask = cp.ones((H, W), dtype=cp.float32)
        mask[r < (H//2 * ratio)] = 0
        
        mask = cp.repeat(mask, self.channel)
        mask = mask.reshape(H, W, self.channel)

        # フィルタ処理
        _G *= mask

        # 位置を戻す
        G[:H//2, :W//2] = _G[H//2:, W//2:]
        G[:H//2, W//2:] = _G[H//2:, :W//2]
        G[H//2:, :W//2] = _G[:H//2, W//2:]
        G[H//2:, W//2:] = _G[:H//2, :W//2]

start_time = time.time()

img_file_path = "../imori.jpg"
process = process_class(img_file_path)
process.dft()
G_orig = process.G.copy()

process.hpf()
process.idft()
G_after = process.G

# パワースペクトル用の処理（GPU配列のまま計算して、最後に .get() する）
G_orig = cp.abs(G_orig)
G_after = cp.abs(G_after)

G_orig = cp.log(1 + G_orig)
G_after = cp.log(1 + G_after)

a = 0
b = 255
c = G_orig.min()
d = G_orig.max()

G_orig = (b - a)/(d - c) * (G_orig - c) + a

# OpenCVのnormalizeを使うため、ここでCPU(NumPy)に変換
G_after_cpu = G_after.get()
G_after_cpu = cv2.normalize(G_after_cpu, None, 0, 255, cv2.NORM_MINMAX)

# 残りのクリップと型変換
G_orig = cp.clip(G_orig, 0, 255).astype(cp.uint8).get()
G_after = cp.clip(G_after, 0, 255).astype(cp.uint8).get()

cv2.imwrite("./result/Question_34.jpg", process.idft_img)
cv2.imwrite("./result/Question_34_G_orig.jpg", G_orig)
cv2.imwrite("./result/Question_34_G_after.jpg", G_after)

# GPUのすべての計算が完全に終了するまで待つ
cp.cuda.Device().synchronize()
end_time = time.time()
print(f"実行時間 : {end_time - start_time:.4} 秒")
