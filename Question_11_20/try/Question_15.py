# ソーベルフィルタを行うプログラムファイル
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gray_img = None
        self.sobel_v_img = None
        self.sobel_h_img = None

    def BGR2GRAY(self):
        if len(self.img.shape) != 3:
            print("画像がカラー画像ではありません")
        else:
            b = self.img[:,:, 0].copy()
            g = self.img[:,:, 1].copy()
            r = self.img[:,:, 2].copy()

            out = 0.216*r + 0.7152*g + 0.0722*b
            out = out.astype(np.uint8)

            self.gray_img = out

    def sobel_filter(self, K_size=3):
        if len(self.gray_img.shape) == 3:
            H, W, C = self.gray_img.shape
        else:
            img = np.expand_dims(self.gray_img, axis=-1)
            H, W, C = img.shape
        
        #ゼロパディング
        pad = K_size // 2
        out = np.zeros((H + pad*2, W + pad*2), dtype=np.float32)
        out[pad:H+pad, pad:W+pad] = self.gray_img.copy().astype(np.float32)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        # 縦方向ソーベルフィルタ
        Kv = [
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]
            ]
        
        Kh = [
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.],
        ]

        for y in range(H):
            for x in range(W):
                out_v[y+pad , x+pad] = np.sum(Kv * tmp[y:y+K_size, x:x+K_size])
                out_h[y+pad , x+pad] = np.sum(Kh * tmp[y:y+K_size, x:x+K_size])
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        self.sobel_v_img = out_v[pad:H+pad, pad:W+pad].astype(np.uint8)
        self.sobel_h_img = out_h[pad:H+pad, pad:W+pad].astype(np.uint8)

img_file_path = "../imori.jpg"
process = process_class(img_file_path)
process.BGR2GRAY()
process.sobel_filter()

cv2.imwrite("./result/Question_15_v.jpg", process.sobel_v_img)
cv2.imwrite("./result/Question_15_h.jpg", process.sobel_h_img)
