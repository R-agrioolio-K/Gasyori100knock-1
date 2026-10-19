# アフィン変換を行うためのプログラムファイル
# アフィン変換とは3×3の行列を用いて画像の変換を行う操作
# ここでは平行移動を行う
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.affine_hor = None
    
    def affin_horizontal(self, a, b, c, d, tx, ty):
        H, W, C = self.img.shape

        out_tmp = np.zeros((H+2, W+2, C), dtype=np.float32)
        out_tmp[1:H+1, 1:W+1] = self.img

        # 新しい画像の型を作成
        # a, dはサイズの拡大・縮小の倍率を表しているため
        # H*d, W*aのようになっている
        H_new = np.round(H*d).astype(np.int32)
        W_new = np.round(W*a).astype(np.int32)
        out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

        # 新しい画像が原画像の参照すべき位置を定める
        # 列番号を格納した2次元配列を作成
        x_new = np.tile(np.arange(W_new), (H_new, 1))
        # 行番号を格納した2次元配列を作成
        y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

        # ここは返還後の画像から逆変換による原画像に対する配列番号割り当ての式を計算
        adbc = a*d - b*c
        x = np.round((d*x_new - b*y_new) / adbc).astype(np.int32) - tx + 1
        y = np.round((-c*x_new + a*y_new) / adbc).astype(np.int32) - ty + 1

        x = np.minimum(np.maximum(x, 0), W+1).astype(np.int32)
        y = np.minimum(np.maximum(y, 0), H+1).astype(np.int32)

        out[y_new, x_new] = out_tmp[y, x]

        out = out[:H_new, :W_new]
        out = out.astype(np.uint8)

        self.affin_hor = out

img_file_path = "../imori.jpg"

process = process_class(img_file_path)
process.affin_horizontal(a=1, b=0, c=0, d=1, tx=30, ty=-30)

cv2.imwrite("./result/Question_28.jpg", process.affin_hor)

        