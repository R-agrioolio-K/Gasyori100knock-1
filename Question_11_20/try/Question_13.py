# MAX-MINフィルタについて実装するプログラムファイル
# MAX-MINフィルタはフィルタ内の画素の最大値と最小値の差を出力するフィルタ
# エッジ検出のフィルタの一種でありグレースケール画像に対してフィルタリングを行う
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gray_img = None
        self.max_min_img = None
    
    def BGR2GRAY(self):
        b = self.img[:,:, 0].copy()
        g = self.img[:,:, 1].copy()
        r = self.img[:,:, 2].copy()

        #グレースケール変換
        out = 0.2126*r + 0.7152*g + 0.0722*b
        out = out.astype(np.uint8)

        self.gray_img = out

    def max_min_filter(self, k_size=3):
        pad = k_size // 2

        # カラー画像に関する処理
        if len(self.img.shape) == 3:
            limg = self.img.copy().astype(np.float32)

            H, W, C = limg.shape
            out = np.zeros([H + pad*2, W + pad*2, C], dtype=np.float32)
            tmp = limg.copy().astype(np.float32)
            out[pad:H+pad, pad:W+pad] = tmp.copy().astype(np.float32)

            # フィルタ処理の開始
            for y in range(H):
                for x in range(W):
                    for c in range(C):
                        out[pad + y, pad + x, c] = np.max(tmp[y:y+k_size, x:x+k_size, c]) - np.min(tmp[y:y+k_size, x:x+k_size, c])
        else:
            limg = self.gray_img.copy().astype(np.float32)

            H, W = limg.shape
            out = np.zeros([H+pad*2, W+pad*2], dtype=np.float32)
            out[pad:pad+H, pad:pad+W] = limg.copy().astype(np.float32)
            tmp = out.copy()

            #フィルタ処理
            for y in range(H):
                for x in range(W):
                    out[y+pad,x+pad] = np.max(tmp[y:y+k_size, x:x+k_size]) - np.min(tmp[y:y+k_size, x:x+k_size])
        
        out = out[pad:H+pad, pad:W+pad]
        self.max_min_img = out.copy().astype(np.uint8)


img_file_path = "../imori.jpg"
process = process_class(img_file_path)
process.BGR2GRAY()
process.max_min_filter()

cv2.imwrite("./result/Question_13.jpg", process.max_min_img)
#cv2.imshow("max_min_filter image", process.max_min_img)
#cv2.waitkey(0)
#cv2.destroyAllWindows()
