# プリューウィットフィルタの処理を行うプログラム
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gray = None
        self.prewitt_h = None
        self.prewitt_v = None
    
    def BGR2GRAY(self):
        b = self.img[:,:, 0].copy()
        g = self.img[:,:, 1].copy()
        r = self.img[:,:, 2].copy()

        out = 0.2126*r + 0.7152*g + 0.0722*b
        self.gray = out.astype(np.uint8)

    def prewitt_filter(self, K_size=3):
        if (self.gray.all()) == None:
            self.BGR2GRAY()
        
        H, W = self.gray.shape

        #ゼロパディング
        pad = K_size // 2
        out = np.zeros((H + pad*2, W + pad*2), dtype=np.float32)
        out[pad:H+pad, pad:W+pad] = self.gray.copy().astype(np.float32)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        #フィルタの定義
        Kv = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
            ])
        
        Kh = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

        #print("type Kh : ", type(Kh))
        for y in range(H):
            for x in range(W):
                out_v[y+pad, x+pad] = np.sum(Kv * tmp[y:y+K_size, x:x+K_size])
                out_h[y+pad, x+pad] = np.sum(Kh * tmp[y:y+K_size, x:x+K_size])
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
        out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)

        self.prewitt_v = out_v.copy()
        self.prewitt_h = out_h.copy()


img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.BGR2GRAY()
process.prewitt_filter()

cv2.imwrite("./result/Question_16_v.jpg", process.prewitt_v)
cv2.imwrite("./result/Question_16_h.jpg", process.prewitt_h)

