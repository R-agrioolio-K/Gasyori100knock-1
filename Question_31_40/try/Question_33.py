# ローパスフィルタを用いたDFTとIDFTを行う
import cv2
import numpy as np
import matplotlib.pyplot as plt

class process_class:
    def __init__(self, img_file_path, K = 128, L = 128, channel = 3):
        self.img = cv2.imread(img_file_path)
        self.K = K
        self.L = L
        self.channel = channel
        self.gray = None
        self.G = None
        self.idft_img = None
    
    def BGR2GRAY(self):
        b = self.img[:,:, 0]
        g = self.img[:,:, 1]
        r = self.img[:,:, 2]

        gray = 0.02126*r + 0.7152*g + 0.0722*b
        self.gray = gray
    
    def dft(self):
        #フーリエ変換による配列を作成
        G = np.zeros((self.L, self.K, self.channel), dtype=np.complex128)

        # 原画像における座標番号を格納した配列を作成する
        H, W, _ = self.img.shape
        y = np.arange(H).repeat(W).reshape(H, -1)
        x = np.tile(np.arange(W), (H, 1))

        # dftの実行
        for c in range(self.channel):
            for l in range(self.L):
                for k in range(self.K):
                    G[l, k, c] = (np.sum(self.img[...,c] 
                                    * np.exp(-2j * np.pi * ((x*k/self.K) + (y*l/self.L)))
                                    / np.sqrt(self.K*self.L)))
        
        self.G = G
    
    def idft(self):
        H, W, _ = self.G.shape
        out = np.zeros((H, W, self.channel), dtype=np.float32)

        # 復元する画像の配列番号が格納された二次元配列を作成
        y = np.arange(H).repeat(W).reshape(H, -1)
        x = np.tile(np.arange(W), (H, 1))

        # idftによる復元
        for c in range(self.channel):
            for l in range(self.L):
                for k in range(self.K):
                    out[l, k, c] = (np.abs(np.sum(self.G[..., c] * 
                    np.exp(2j * np.pi * (x*k/W + y*l/H))) / np.sqrt(W*H)))
        
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        
        self.idft_img = out
    
    # ローパスフィルタの実装
    def lpf(self, ratio=0.5):
        H, W, _ = self.G.shape
        G = self.G

        # 中心に値が大きいものが入るように位置を変更する
        # 普通に手作業で位置を変更しているG => _G(変更後)
        _G = np.zeros_like(G)
        _G[:H//2, :W//2] = G[H//2:, W//2:]
        _G[:H//2, W//2:] = G[H//2:, :W//2]
        _G[H//2:, :W//2] = G[:H//2, W//2:]
        _G[H//2:, W//2:] = G[:H//2, :W//2]

        y = np.arange(H).repeat(W).reshape(H, -1)
        x = np.tile(np.arange(W), (H, 1))

        # フィルターを作成する
        # 中心を原点とする座標を作り出すための処理
        _x = x - W//2
        _y = y - H//2
        # 中心からの距離が格納された2次元配列rを作成する
        r = np.sqrt(_x**2 + _y**2)
        # np.onesはnp.ones((行, 列), dtype=型)となる全ての値が1の配列を作成
        mask = np.ones((H, W), dtype=np.float32)
        # maskの配列に対して[r > (H // 2 *ratio) ]の条件が成立する配列番号に対して
        # 値を変更するのはmaskに格納されている値だが、条件式を確認するのはmaskに格納されている値
        # つまりmaskとrが同じ形状の配列でしか成立しないs
        mask[r > (H // 2 * ratio)] = 0

        mask = np.repeat(mask, self.channel).reshape(H, W, self.channel)

        # フィルタリング
        _G *= mask

        # 元の位置に戻す
        G[:H//2, :W//2] = _G[H//2:, W//2:]
        G[:H//2, W//2:] = _G[H//2:, :W//2]
        G[H//2:, :W//2] = _G[:H//2, W//2:]
        G[H//2:, W//2:] = _G[:H//2, :W//2]

        self.G = G


img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.dft()
# コピーとして配列を新たなメモリ領域で作成しないと
# ポインタが渡されてしまう
G_orig = process.G.copy()

process.lpf()
process.idft()
G_after = process.G


# 原画像のパワースペクトルを作成するべく
# Gに対して絶対値 + 対数 + 正規化
G_orig = np.abs(G_orig)
G_orig = np.log(1+G_orig)
a = 0
b = 255
c = G_orig.min()
d = G_orig.max()
G_orig = (b - a)/(d - c) * (G_orig - c) + a
G_orig = np.clip(G_orig, 0, 255).astype(np.uint8)

# ローパスフィルタあとのパワースペクトルを作成するべく
# G_afterに対して絶対値 + 対数 + 正規化
G_after = np.abs(G_after)
G_after = np.log(1+G_after)
a = 0
b = 255
c = G_after.min()
d = G_after.max()
G_after = (b - a)/(d - c) * (G_after - c) + a
G_after = np.clip(G_after, 0, 255).astype(np.uint8)

cv2.imwrite("./result/Question_33.jpg", process.idft_img)
cv2.imwrite("./result/Question_33_G_orig.jpg", G_orig)
cv2.imwrite("./result/Question_33_G_after.jpg", G_after)


