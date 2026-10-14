#LoGフィルタを実装するプログラム
"""
LoGフィルタはLaplacian of Gaussianである。
ガウシアンフィルタで画像を平滑化した後にラプラシアンフィルタで輪郭を取り出すフィルタ
ラプラシアンフィルタは二次微分をとるのでノイズが強調されるのを防ぐため
あらかじめGaussianフィルタでノイズを抑える
"""
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gray = None
        self.LoG = None
    
    def BGR2GRAY(self):
        B = self.img[:,:, 0]
        G = self.img[:,:, 1]
        R = self.img[:,:, 2]

        gray = 0.2126*R + 0.7152*G + 0.0722*B
        self.gray = gray
    
    def LoG_filter(self, K_size=5, sigma=3):
        if (self.gray.all()) == None:
            self.BGR2GRAY()
        
        H, W = self.gray.shape

        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad:H+pad, pad:W+pad] = self.gray.copy().astype(np.float32)
        tmp = out.copy()

        #LoGカーネルの定義
        # ガウス関数をx方向とy方向に二次微分した係数を計算する
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y+pad, x+pad] = (x**2 + y**2 - sigma**2) * np.exp(-(x**2 + y**2) / (2 * (sigma ** 2)))
        
        K /= (2 * np.pi * (sigma ** 6))
        K /= K.sum()

        print("K : \n",K) #for debug
        print("K size : ",K.shape)  #for debug

        #フィルタ処理
        for y in range(H):
            for x in range(W):
                out[pad+y, pad+x] = np.sum(K * tmp[y:y+K_size, x:x+K_size])
        
        out = np.clip(out, 0, 255)
        out = out[pad:H+pad, pad:W+pad]

        self.LoG = out.astype(np.uint8)
    
img_file_path = "../imori_noise.jpg"
process = process_class(img_file_path)
process.BGR2GRAY()
process.LoG_filter()

cv2.imwrite("./result/Question_19.jpg", process.LoG)