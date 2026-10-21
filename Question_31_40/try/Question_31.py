# アフィン変換によるスキュー画像の作成を行う
# スキュー画像は斜め方向に伸ばしたような画像のこと
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.affin_img = None
    
    def affin(self, dx=30, dy=30):
        H, W, C = self.img.shape

        a = 1.
        b = dx / H  # 変換後の画像におけるx座標がy座標に対する倍率の影響を受ける
        c = dy / W  # 変換後の画像におけるy座標が参照すべき原画像のy座標に対して変換後の画像におけるx座標の相対位置の影響を受ける
        d = 1.
        tx = 0.
        ty = 0.

        _img = np.zeros((H+2, W+2, C), dtype=np.float32)

        # 原画像を作成した一時保存用の配列に挿入する
        _img[1:H+1, 1:W+1] = self.img

        # アフィン変換後の画像の形状を作成する
        H_new = np.ceil(dy + H).astype(int)
        W_new = np.ceil(dx + W).astype(int)
        out = np.zeros((H_new, W_new, C), dtype=np.float32)

        # アフィン変換あとの画像における配列番号(行、列)を持つ配列を作成する
        x_new = np.tile(np.arange(W_new), (H_new, 1))
        y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

        # 逆行列による元画像における参照すべき座標を算出
        adbc = a*d - b*c
        x = np.round((d*x_new - b*y_new) / adbc).astype(int) - tx + 1
        y = np.round((-c*x_new + a*y_new) / adbc).astype(int) - ty + 1

        x = np.minimum(np.maximum(x, 0), W+1).astype(int)
        y = np.minimum(np.maximum(y, 0), H+1).astype(int)

        out[y_new, x_new] = _img[y, x]
        out = out.astype(np.uint8)
        
        self.affin_img = out

img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.affin()
cv2.imwrite("./result/Question_31.jpg", process.affin_img)