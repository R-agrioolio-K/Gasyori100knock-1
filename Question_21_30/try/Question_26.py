# バイリニア補間
# バイリニア補間は周辺の4画素に距離に応じた重みをつけることで
import cv2
import numpy as np
import matplotlib.pyplot as plt

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.Bilinear = None
    
    def BI_interpolate(self, ax=1., ay=1.):
        H, W, C = self.img.shape

        aH = int(ay * H)
        aW = int(ax * W)

        # 行番号を持った縦aH行, 横aW列の2次元配列を作成
        # arangeは0からaH - 1までの数字を生成
        # repeatは各要素を連続でaW回繰り返す
        # -1はndarrayクラスのもつ多次元配列の形状における自動計算
        y = np.arange(aH).repeat(aW).reshape(aH, -1)
        
        # 列番号を持った縦aH行, 横aW列の2次元配列を作成
        # np.tile(np.arange(aW), (aH, 1))について
        # np.arange(aW)により0, 1, ... , aW-1までを
        # aH行分，1列分の2次元配列として作成
        x = np.tile(np.arange(aW), (aH, 1))

        # x, yのすべての要素に対して倍率をかけることで出力画像のそれぞれが参照するべき位置を指定
        y = (y / ay)
        x = (x / ax)

        # np.floorは-無限方向へ少数以下切り捨て
        # -1.2 -> -2になるということ
        ix = np.floor(x).astype(np.uint8)
        iy = np.floor(y).astype(np.uint8)

        # np.minimumは各要素の小さい値を選択する
        # 参照する座標に関して原画像は縦横H-1, W-1までしか存在しないため
        # ixとiyは座標を表す
        ix = np.minimum(ix, W-2)
        iy = np.minimum(iy, H-2)
        print("ix : ",ix[0,0:11])

        # 変形後の画素の座標が元の画像の参照すべき座標からどれだけ離れているかを表す
        # ix.shape = (192, 192)になっている
        # dx, dyは少数になりこれが各画素における重みを決定する
        dx = x - ix
        dy = y - iy
        
        # チャンネル方向への次元拡大3回繰り返す
        dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
        dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

        # 形状（Dimensions）とデータ型を確認
        #print("dx shape:", dx.shape, "dtype:", dx.dtype)
        print("img shape:", self.img.shape)

        # 値の最小値・最大値を確認（0~1に収まっているかなど）
        print(f"dx range: min={dx.min():.4f}, max={dx.max():.4f}")
        print(f"iy range: min={iy.min()}, max={iy.max()}")

        # for文を使わずに処理するための記述がself.img[iy, ix]
        # 行と列それぞれで
        out = (
            (1-dx) * (1-dy) * self.img[iy, ix] 
            + dx * (1 - dy) * self.img[iy, ix+1] 
            + (1 - dx)* dy * self.img[iy+1, ix] 
            + dx * dy * self.img[iy+1, ix+1]
            )
        print("1 - dx shape : ", (1-dy).shape)
        print("self.img shape : ", self.img.shape)
        print("self.img[iy, ix] shape : ",self.img[iy, ix].shape)
        print("self.img[iy, ix+1] shape : ",self.img[iy, ix+1].shape)

            
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        self.Bilinear = out

img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.BI_interpolate(ax = 0.7, ay = 0.8)

cv2.imwrite("./result/Question_26.jpg", process.Bilinear)


