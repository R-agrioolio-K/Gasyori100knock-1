# アフィン変換を実行するプログラムファイル
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.affin_img = None
    
    def affin(self, a, b, c, d, tx, ty):
        H, W, C = self.img.shape
        print("W : ",W)

        # 原画像の一時保管
        tmp_img = np.zeros((H+2, W+2, C), dtype= np.float32)
        tmp_img[1:H+1, 1:W+1] = self.img

        # 新しい画像のサイズを計算する
        H_new = np.round(H*d).astype(int)
        W_new = np.round(W*a).astype(int)
        print("H_new : ",H_new)
        print("W_new : ",W_new)
         
        # 変換あとの画像を0配列で作成する
        # 少数などがかかわるため配列からはみ出してエラーにならないように大きいサイズにする
        out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

        # 新しい画像の配列番号を格納する
        y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
        x_new = np.tile(np.arange(W_new), (H_new, 1))

        # 原画像の参照すべき位置を計算する
        # 参照位置に対して末尾に +1 を入れているのはtmp_imgが縦横に対して両端に余白が1画素含んでいるから
        adbc = a*d - b*c
        x = np.round((d*x_new - b*y_new) / adbc).astype(int) - tx + 1
        y = np.round((-c*x_new + a*y_new)/adbc).astype(int) - ty + 1

        # 原画像の座標にない場合の処理
        x = np.minimum(np.maximum(x, 0), W+1).astype(int)
        y = np.minimum(np.maximum(y, 0), H+1).astype(int)

        # 最後はシンプル出力用配列の各画素に対して原画像のどの画素を参照するかのマップにより終わり
        out[y_new, x_new] = tmp_img[y, x]

        out = out[:H_new, :W_new]
        out = out.astype(np.uint8)

        self.affin_img = out

img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.affin(a = 1.3, b=0, c=0, d=0.8, tx=0, ty=0)
cv2.imwrite("./result/Question_29_01.jpg", process.affin_img)

process.affin(a = 1.3, b=0, c=0, d=0.8, tx=30, ty=-30)
cv2.imwrite("./result/Question_29_02.jpg", process.affin_img)