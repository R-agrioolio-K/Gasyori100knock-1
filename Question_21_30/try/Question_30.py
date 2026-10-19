# アフィン変換による画像の回転を行う
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.affin_img = None
    
    def affin_method(self, a, b, c, d, tx, ty):
        H, W, C = self.img.shape

        img_temp = np.zeros((H+2, W+2, C), dtype=np.float32)
        img_temp[1:H+1, 1:W+1] = self.img

        # 変換後の画像の形状を算出する
        # 回転の場合はサイズはそのままにする
        H_new = np.round(H).astype(int)
        W_new = np.round(W).astype(int)
        out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

        # 変換後の画像の行、列の番号を格納する2次元配列を作成
        y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
        x_new = np.tile(np.arange(W_new), (H_new, 1))

        # 変換後の画像が参照すべき原画像の座標を計算する
        adbc = a*d - b*c
        x = np.round((d*x_new - b*y_new) / adbc).astype(int) - tx + 1
        y = np.round((-c*x_new + a*y_new) / adbc).astype(int) - ty + 1

        # 画像がなるべく収まるように調整する
        # 中央がどれだけずれているかを算出する
        dcx = (x.max() + x.min()) // 2 - W // 2
        dcy = (y.max() + y.min()) // 2 - H // 2
        print(f"H : {H}, W : {W}")
        print(f"x.max : {x.max()}, x.min : {x.min()}")
        print(f"y.max : {y.max()}, y.min : {y.min()}")

        x -= dcx
        y -= dcy

        x = np.clip(x, 0, W+1)
        y = np.clip(y, 0, H+1)

        out[y_new, x_new] = img_temp[y, x]
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = out[:H_new, :W_new]

        self.affin_img = out

img_file_path = "../imori.jpg"
process = process_class(img_file_path)

A = 30.
theta = -np.pi * A / 180.

process.affin_method(a=np.cos(theta), b=-np.sin(theta), c=np.sin(theta), d=np.cos(theta)
,tx = 0, ty=0)

cv2.imwrite("./result/Question_30.jpg", process.affin_img)

