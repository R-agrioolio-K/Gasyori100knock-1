# Embossフィルタを実装する
# Embossフィルタは会陰各部分を浮き出しにするフィルタ
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gray = None
        self.emboss = None
    
    def BGR2GRAY(self):
        if len(self.img.shape) == 3:
            b = self.img[:,:, 0]
            g = self.img[:,:, 1]
            r = self.img[:,:, 2]

            gray = 0.2126*r + 0.7152*g + 0.0722*b

            self.gray = gray
        else:
            self.gray = img.copy()
    
    def emboss_filter(self, K_size=3):
        H, W = self.gray.shape

        #ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = self.gray.copy().astype(np.float32)
        tmp = out.copy()

        #フィルタ定義
        K = np.array(
            [
                [-2., -1., 0.],
                [-1., 1., 1.],
                [0., 1., 2.]
            ]
        )
        #エンボスフィルタ処理
        for y in range(H):
            for x in range(W):
                out[y+pad, x+pad] = np.sum(K * tmp[y:y+K_size, x:x+K_size])
        
        out = np.clip(out, 0, 255)
        out = out[pad:H+pad, pad:W+pad].astype(np.uint8)

        self.emboss = out

img_file_path = "../imori.jpg"
process = process_class(img_file_path)
process.BGR2GRAY()
process.emboss_filter()

cv2.imwrite("./result/Question_18.jpg", process.emboss)

