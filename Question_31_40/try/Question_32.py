# フーリエ変換を行うプログラム
"""
二次元離散フーリエ変換(DFT)を実装し，imori.jpgをグレースケール化したものの
周波数のパワースペクトルを表示する
また，逆二次元離散フーリエ変換(IDFT)で画像を復元する
"""
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path, K=128, L=128):
        self.img = cv2.imread(img_file_path)
        self.K = K
        self.L = L
        self.gray_img = None
        self.dft_img = None
        self.idft_img = None
    
    def RGB2GRAY(self):
        b = self.img[:, :, 0]
        g = self.img[:, :, 1]
        r = self.img[:,:, 2]

        out = 0.0114*b + g*0.587 + 0.2989*r

        self.gray_img = out
    
    def dft(self):
        H, W, C = self.img.shape

        # DFT(フーリエ変換)係数Gの設定
        G = np.zeros((self.L, self.K, C), dtype=np.complex128)

        # prepare processed index corresponding to original image positions
        # 原画像の位置を格納する配列を作成
        y = np.arange(H).repeat(W).reshape(H, -1)
        x = np.tile(np.arange(W), (H, 1))

        # dft変換
        for c in range(C):
            for l in range(self.L):
                for k in range(self.K):
                    # 元画像の位置情報に基づいた処理はexpの中身部分にあたる
                    G[l, k, c] = (np.sum(self.img[..., c] 
                    * np.exp(-2j * np.pi * ((x*k / self.K) + (y*l / self.L)))) / np.sqrt(self.K*self.L))

        print("G(変換された値) : \n",G[:3,:5,0])
        self.dft_img = G

    def idft(self):
        H, W, C = self.dft_img.shape
        out = np.zeros((H, W, C), dtype=np.float32)

        y = np.arange(H).repeat(W).reshape(H, -1)
        x = np.tile(np.arange(W), (H, 1))

        # 逆フーリエ変換
        for c in range(C):
            for l in range(self.L):
                for k in range(self.K):
                    out[l, k, c] = (np.abs(np.sum(self.dft_img[..., c] 
                    * np.exp(2j * np.pi * ((x*k / W)+(y*l / H))))) / np.sqrt(W*H))
        
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        self.idft_img = out


img_file_path = "../imori.jpg"
process = process_class(img_file_path, K=128, L=128)
process.dft()
process.idft()

# 1. 複素数配列の絶対値（振幅スペクトル）を計算する
magnitude = np.abs(process.dft_img)

# 2. （推奨）ダイナミックレンジが広すぎるため、対数変換を行って見やすくする
# 0での対数計算を防ぐために +1 を加える
log_spectrum = np.log(1 + magnitude)

# 3. 0〜255 の範囲に正規化（スケーリング）する
#normalized = cv2.normalize(log_spectrum, None, 0, 255, cv2.NORM_MINMAX)
a = 0
b = 255
c = log_spectrum.min()
d = log_spectrum.max()
normalized = (b - a) / (d - c) * (log_spectrum - c) + a
normalized[normalized < a] = a
normalized[normalized > b] = b

# 4. uint8型に変換して保存する
ps_img = np.clip(normalized, 0, 255).astype(np.uint8)


cv2.imwrite("./result/Question_32_ps.jpg", ps_img)
cv2.imwrite("./result/Question_32.jpg", process.idft_img)
