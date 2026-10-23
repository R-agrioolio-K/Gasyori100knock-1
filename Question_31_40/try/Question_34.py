# DFTしたものをハイパスフィルタにより
# 高周波成分のみを通す
import cv2
import numpy as np
import time

class process_class:
    def __init__(self, img_file_path, K = 128, L = 128, channel = 3):
        self.img = cv2.imread(img_file_path)
        self.G = None
        self.idft_img = None
        self.K = K
        self.L = L
        self.channel = channel
    
    def dft(self):
        H, W, _ = self.img.shape

        # 変換後の配列番号を格納したx, y配列を作成する
        y = np.arange(H).repeat(W).reshape(H, -1)
        x = np.tile(np.arange(W), (H, 1))
        out = np.zeros((self.L, self.K, self.channel), dtype=np.complex128)


        # dftによる変換
        for c in range(self.channel):
            for l in range(self.L):
                for k in range(self.K):
                    out[l, k, c] = (np.sum(self.img[..., c] 
                    * np.exp(-2j * np.pi *((k*x/self.K) + (l*y/self.L))))/np.sqrt(H*W))
        
        self.G = out
    
    def idft(self):
        H, W, _ = self.G.shape
        out = np.zeros((H, W, self.channel), dtype=np.float32)

        # 変換後の配列番号を格納したx, yの配列を作成する
        y = np.arange(H).repeat(W).reshape(H, -1)
        x = np.tile(np.arange(W), (H, 1))
        out = np.zeros((H, W, self.channel), dtype=np.float32)

        for c in range(self.channel):
            for l in range(self.L):
                for k in range(self.K):
                    out[l, k, c] = (np.abs(np.sum(self.G[...,c] 
                    * np.exp(2j * np.pi * (x*k/W + y*l/H)))) / np.sqrt(W*H))
        
        # クリッピング
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        self.idft_img = out
    
    def hpf(self, ratio = 0.1):
        H, W, _ = self.G.shape
        G = self.G

        # 位置を移動する
        _G = np.zeros_like(G)
        _G[:H//2, :W//2] = G[H//2:, W//2:]
        _G[:H//2, W//2:] = G[H//2:, :W//2]
        _G[H//2:, :W//2] = G[:H//2, W//2:]
        _G[H//2:, W//2:] = G[:H//2, :W//2]

        # マスクを作成する
        x = np.tile(np.arange(W), (H, 1))
        y = np.arange(H).repeat(W).reshape(H, -1)
        _x = x - W // 2
        _y = y - H // 2
        r = np.sqrt(_x**2 + _y**2)
        mask = np.ones((H, W), dtype=np.float32)
        mask[r < (H//2 * ratio)] = 0
        #チャネル方向への拡大を行う
        # この時reshapeを行わない場合どのような形状になっている?
        mask = np.repeat(mask, self.channel)
        print("hpf np.repeat(mask, channel) shape : ",mask.shape)
        mask = mask.reshape(H, W, self.channel)

        # フィルタ処理
        _G *= mask

        #位置を戻す
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

# パワースペクトルを見えるようにするための処理
# 絶対値 + 対数 + 正規化 + クリップ

# 絶対値処理
G_orig = np.abs(G_orig)
G_after = np.abs(G_after)

# 対数処理
G_orig = np.log(1+G_orig)
G_after = np.log(1+G_after)

#正規化のためのパラメータ設定
a = 0
b = 255
c = G_orig.min()
d = G_orig.max()

# 正規化処理
G_orig = (b - a)/(d - c) * (G_orig - c) + a

# cv2を使った正規化処理
G_after = cv2.normalize(G_after, None, 0, 255, cv2.NORM_MINMAX)

# クリップと型の変換
G_orig = np.clip(G_orig, 0, 255).astype(np.uint8)
G_after = np.clip(G_after, 0, 255).astype(np.uint8)

cv2.imwrite("./result/Question_34.jpg", process.idft_img)
cv2.imwrite("./result/Question_34_G_orig.jpg", G_orig)
cv2.imwrite("./result/Question_34_G_after.jpg", G_after)

end_time = time.time()
print(f"実行時間 : {end_time - start_time:.4f} 秒")

        