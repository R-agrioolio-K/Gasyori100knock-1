# バイキュービック補間を行うプログラム
# Bi-cubic補間はBilinear補間が4近傍だったのに対して16近傍で行う
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.bi_cubic = None
    
    # bi-cubic 重み
    def weight(self, t):
        a = -1.
        at = np.abs(t)
        w = np.zeros_like(t)
        ind = np.where(at <= 1)
        w[ind] = ((a+2) * np.power(at, 3) - (a+3)*np.power(at, 2) + 1)[ind]
        ind = np.where((at > 1)&(at <= 2))
        w[ind] = (a*np.power(at, 3) - 5*a*np.power(at, 2) + 8*a*at - 4*a)[ind]
        
        return w

    def Bi_cubic(self, ax=1, ay=1):
        H, W, C = self.img.shape

        aH = int(H * ay)
        aW = int(W * ax)

        y = np.arange(aH).repeat(aW).reshape(aH, -1)
        x = np.tile(np.arange(aW), (aH, 1))
        y = (y / ay)
        x = (x / ax)

        # 原画像に対して参照すべき位置を取得する
        ix = np.floor(x).astype(np.int32)
        iy = np.floor(y).astype(np.int32)

        ix = np.minimum(ix, W-1)
        iy = np.minimum(iy, H-1)

        # 原画像とサイズ変更後の画像における各画素の距離の差を重みとする
        dx2 = x - ix
        dy2 = y - iy
        dx1 = dx2 + 1
        dy1 = dy2 + 1
        dx3 = dx2 - 1
        dy3 = dy2 - 1
        dx4 = dx3 - 1
        dy4 = dy3 - 1

        dxs = [dx1, dx2, dx3, dx4]
        dys = [dy1, dy2, dy3, dy4]

        
        w_sum = np.zeros((aH, aW, C), dtype=np.float32)
        out = np.zeros((aH, aW, C), dtype=np.float32)

        # 補間処理
        # 生成される画素に対して参照すべき座標は相対位置-1 ~ +2までで定義されるため
        # range(-1, 3)を用いたfor文を用いる
        for j in range(-1, 3):
            for i in range(-1, 3):
                # 高度なインデックス参照により原画像の-1から順に参照を行う
                ind_x = np.minimum(np.maximum(ix + i, 0), W-1)
                ind_y = np.minimum(np.maximum(iy + j, 0), H-1)

                # 参照する画素のが生成する画素からの位置の違いによる重みフィルタを作成
                wx = self.weight(t = dxs[i+1])
                wy = self.weight(t = dys[j+1])
                wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1) # 3チャンネル分に拡張する
                wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1) # 3チャンネル分に拡張する

                w_sum += wx * wy
                out += wx * wy * self.img[ind_y, ind_x]
        
        out /= w_sum
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        self.bi_cubic = out

img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.Bi_cubic(ax=1.8, ay=1.8)

cv2.imwrite("./result/Question_27.jpg", process.bi_cubic)