# ラプラシアンフィルタ(2次微分フィルタ)
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gray = None
        self.laplacian = None
    
    def BGR2GRAY(self, K_size=3):
        b = self.img[:,:, 0]
        g = self.img[:,:, 1]
        r = self.img[:,:, 2]

        out = 0.2126*r + 0.7152*g + 0.0722*b
        self.gray = out.astype(np.uint8)
    
    def laplacian_filter(self, K_size=3):
        if (self.gray.all()) == None:
            self.BGR2GRAY()
        
        H, W = self.gray.shape

        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = self.gray.copy().astype(np.float32)
        tmp = out.copy()

        #カーネルの定義
        K = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]

        #ラプラシアンフィルタ処理
        for y in range(H):
            for x in range(W):
                out[y+pad, x+pad] = np.sum(K * tmp[y:y+K_size, x:x+K_size])
        
        out = np.clip(out, 0, 255)
        out = out[pad:H+pad, pad:W+pad].astype(np.uint8)

        self.laplacian = out

img_file_path = "../imori.jpg"
process = process_class(img_file_path)
process.BGR2GRAY()
process.laplacian_filter()

cv2.imwrite("./result/Question_17.jpg", process.laplacian)
