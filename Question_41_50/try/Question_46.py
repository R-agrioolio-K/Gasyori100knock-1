# Hough変換による直線検出
# Hough逆変換
# これによりHough変換による直線検出が終了になる
import cv2 
import numpy as np

class canny_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
    
    def BGR2GRAY(self, img):
        b = img[:,:, 0].copy().astype(np.float16)
        g = img[:,:, 1].copy().astype(np.float16)
        r = img[:,:, 2].copy().astype(np.float16)

        gray = 0.2126*r + 0.7152*g + 0.0722*b
        gray = gray.astype(np.uint8)

        return gray
    
    def gaussian_filter(self, img, K_size=3, sigma=1.3):
        # カラー画像でも処理できるように分岐処理
        if len(img.shape) == 3:
            H, W, C = img.shape
            gray = False
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
            gray = True
        
        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float16)
        out[pad:H+pad, pad:W+pad] = img.copy().astype(np.float16)

        # ガウシアンフィルタの準備
        K = np.zeros((K_size, K_size), dtype=np.float16)
        for x in range(-pad, -pad+K_size):
            for y in range(-pad, -pad+K_size):
                K[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (2*sigma*sigma))
        
        K /= (2*np.pi*sigma*sigma)
        K /= K.sum()

        # 原画像を直接変換しないように一時保存
        tmp = out.copy()

        # フィルタ処理
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad+y, pad+x, c] = np.sum(K*tmp[y:y+K_size, x:x+K_size, c])
        
        out = np.clip(out, 0, 255).astype(np.uint8)
        if gray:
            out = out[:,:,0]
        return out
    
    # ソーベルフィルタ
    def sobel_filter(self, img, K_size=3):
        # 元画像のサイズ取得
        # len(img.shape) = 2の時H, W, _ = img.shapeはエラーになる
        H, W = img.shape

        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float16)
        out[pad:H+pad, pad:W+pad] = img.copy().astype(np.float16)

        # フィルタの作成
        Kv = [
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1]
        ]
        Kh = [
            [1., 0., -1.],
            [2., 0., -2],
            [1., 0., -1.]
        ]
        out_v = out.copy()
        out_h = out.copy()

        tmp = out.copy()

        for y in range(H):
            for x in range(W):
                out_v[pad+y, pad+x] = np.sum(Kv*(tmp[y:y+K_size, x:x+K_size]))
                out_h[pad+y, pad+x] = np.sum(Kh*(tmp[y:y+K_size, x:x+K_size]))
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)
        
        out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
        out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)

        return out_v, out_h

    # エッジ強度の取得
    def get_edge_angle(self, fx, fy):
        # エッジ強度取得
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        edge = np.clip(edge, 0, 255)

        #fx = np.maximum(fx, 1e-10)

        angle = np.arctan2(fy, fx)

        return edge, angle
    
    # 角度の量子化
    def angle_quantization(self, angle):
        angle = angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

        return _angle
    
    def non_maximum_suppression(self, angle, edge):
        H, W = angle.shape
        _edge = edge.copy()

        for y in range(H):
            for x in range(W):
                if angle[y, x] == 0:
                    dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                elif angle[y, x] == 45:
                    dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                elif angle[y, x] == 90:
                    dx1, dy1, dx2, dy2  = 0, -1, 0, 1
                elif angle[y, x] == 135:
                    dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                if x == 0:
                    dx1 = max(dx1, 0)
                    dx2 = max(dx2, 0)
                if x == W-1:
                    dx1 = min(dx1, 0)
                    dx2 = min(dx2, 0)
                if y == 0:
                    dy1 = max(dy1, 0)
                    dy2 = max(dy2, 0)
                if y == H-1:
                    dy1 = min(dy1, 0)
                    dy2 = min(dy2, 0)
                if max(max(edge[y, x], edge[y+dy1, x+dx1]), edge[y+dy2, x+dx2]) != edge[y,x]:
                    _edge[y, x] = 0
        
        return _edge
    
    def hysterisis(self, edge, HT=100, LT=30):
        H, W = edge.shape

        # 閾値処理
        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        _edge = np.zeros((H+2, W+2), dtype=np.float32)
        _edge[1:H+1, 1:W+1] = edge

        # 近傍補完
        # フィルタの定義
        nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float16)

        for y in range(1, H+2):
            for x in range(1, W+2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                # 3×3の範囲における中心画素以外の最大値を取得
                if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0
        
        edge = _edge[1:H+1, 1:W+1]

        return edge

    def all_run(self):
        # グレースケール化
        gray = self.BGR2GRAY(self.img)
        # ガウシアンフィルタ
        gaussian = self.gaussian_filter(gray, K_size=5, sigma=1.4)
        # ソーベルフィルタ
        fy, fx = self.sobel_filter(gaussian, K_size=3)
        # エッジ強度と角度の取得
        edge, angle = self.get_edge_angle(fx, fy)
        # 角度の量子化
        angle = self.angle_quantization(angle)
        # 細線化
        edge = self.non_maximum_suppression(angle, edge)
        # ヒステリシス変換
        out = self.hysterisis(edge, 100, 30)

        return out
    
class Hough_class:
    def __init__(self, edge, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.edge = edge
    
    # cosとsinを規定とした空間に変換した後
    # 投票を行う
    def voting(self, edge):
        H, W = edge.shape
        
        # 直線と角度におけるスライドの定義
        drho = 1
        dtheta = 1

        # 直線距離の最大値の取得
        rho_max = np.ceil(np.sqrt(H**2 + W**2)).astype(np.int32)

        # houghテーブルの定義
        hough = np.zeros((rho_max*2, 180), dtype=np.int32)

        # edgeのインデックス番号を取得
        ind = np.where(edge==255)

        # Hough変換
        for y, x in zip(ind[0], ind[1]):
            for theta in range(0, 180, dtheta):
                # 度数法からラジアンへの変換
                t = np.pi / 180 * theta
                # cos, sinへの射影を行っている
                # 変換の心臓部分
                rho = int(x * np.cos(t) + y * np.sin(t))

                #投票を行う
                hough[rho + rho_max, theta] += 1

        return hough
    
    # non_maximum_suppression(非最大値抑制)
    def non_maximum_suppression(self, hough):
        # houghテーブルにおける行を取得している
        rho_max, _ = hough.shape

        # 非最大値抑制
        for y in range(rho_max):
            for x in range(180):
                # 8近傍における走査を行う
                # 配列番号のはみ出しを防ぐための処理
                x1 = max(x-1, 0)
                x2 = min(x+2, 180)
                y1 = max(y-1, 0)
                y2 = min(y+2, rho_max-1)
                # 注目画素
                if (np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0):
                    pass
                else:
                    hough[y,x] = 0

        return hough

    def inverse_hough(self, hough, img):   
        H, W, _ = img.shape
        rho_max, _ = hough.shape     

        out = img.copy()

        # x, yにhoughテーブルにおけるインデックス番号を取得する
        # ravelは一次元化を行う
        # [::-1]で配列の順番を逆にする
        # [:30]は配列の最初から30番までを取得する
        # 1次元配列に直した後のインデックス番号を取得している
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        ind_y = ind_x.copy()
        # 180で除算した際のあまりを求めることで列番号(角度)を取得
        thetas = ind_x % 180
        # 180で除算した結果の整数部分を取得することで行番号(直線の長さを取得)
        # -rho_max / 2を行うことで2倍の長さではなく本来の長さにしている
        rhos = ind_y // 180 - rho_max / 2

        for theta, rho in zip(thetas, rhos):
            # 度数からラジアンに変換する
            t = np.pi / 180. * theta
            """
            rho = x*cos(t) + y*sin(t)
            y*sin(t) = rho - x*cos(t)
            y = (rho / sin(t)) - ((x*cos(t))/sin(t))
            同様に
            x = (rho / cos(t)) - ((y*sin(t))/cos(t))
            """
            # houghテーブルから(x, y)座標への変換
            # x座標を固定してy座標を求める
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out[y, x] = [0, 0, 255]
            # y座標を固定してx座標を求める
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [0, 0, 255]
        
        out = out.astype(np.uint8)

        return out
    
    def all_run(self):
        # 投票を行う
        hough = self.voting(self.edge)
        # non_maximum_suppression(非極大値の抑制)
        hough = self.non_maximum_suppression(hough)
        # 逆Hough変換
        out = self.inverse_hough(hough, self.img)

        return out.astype(np.uint8)

img_file_path = "../thorino.jpg"
# キャニーエッジ検出
canny_ins = canny_class(img_file_path)
edge = canny_ins.all_run()

# Hough変換
hough_ins = Hough_class(edge, img_file_path)
result = hough_ins.all_run()

cv2.imwrite("./result/Question_46.jpg", result)

    

