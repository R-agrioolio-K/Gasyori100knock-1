# Hough変換・直線検出
# Hough変換による直線の検出を行う
# Hough変換とは座標を直行座標から極座標に変換することにより数式に沿って直線など一定の形状を検出する手法
# ある直線状の点では極座標に変換すると一定のr, tにおいて交わる。
# その点が検出すべき直線を表すパラメータでありこのパラメータを逆変換すると直線の方程式を求めることができる
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Canny_process:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.result = None
    
    def BGR2GRAY(self, img):
        b = img[:,:, 0].copy()
        g = img[:,:, 1].copy()
        r = img[:,:, 2].copy()

        out = 0.2126*r + 0.7152*g + 0.0722*b
        out = out.astype(np.uint8)

        return out
    
    def gaussian_filter(self, img, K_size=3, sigma=1.3):

        if len(img.shape) == 3:
            H, W, C = img.shape
            gray = False
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
            gray = True
        
        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float32)

        ## フィルタの準備
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for x in range(-pad, -pad+K_size):
            for y in range(-pad, -pad+K_size):
                K[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (2*sigma*sigma))
        
        K /= (2*np.pi*sigma*sigma)
        K /= K.sum()

        tmp = out.copy()

        # フィルタ処理
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[y+pad, x+pad, c] = np.sum(K*tmp[y:y+K_size, x:x+K_size, c])
        
        out = np.clip(out, 0, 255)
        out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

        if gray:
            out = out[..., 0]
        
        return out
    
    def sobel_filter(self, img, K_size=3):
        H, W = img.shape

        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float32)
        tmp = out.copy()
        
        # 縦方向エッジと横方向エッジのそれぞれの結果を保存するための配列を定義
        out_v = out.copy()
        out_h = out.copy()

        # 縦方向と横方向のソーベルフィルタの定義
        Kv = [
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]
        ]
        Kh = [
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
        ]

        # フィルタ処理
        for y in range(H):
            for x in range(W):
                out_v[pad+y, pad+x] = np.sum(Kv*(tmp[y:y+K_size, x:x+K_size]))
                out_h[pad+y, pad+x] = np.sum(Kh*(tmp[y:y+K_size, x:x+K_size]))
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad:H+pad, pad:W+pad].astype(np.uint8)
        out_h = out_h[pad:H+pad, pad:W+pad].astype(np.uint8)

        return out_v, out_h
    
    def get_edge_angle(self, fx, fy):
        # エッジ強度を取得する
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        edge = np.clip(edge, 0, 255)

        # 最小値がゼロの場合のゼロ除算を防ぐための置き換え
        fx = np.maximum(fx, 1e-10)

        # エッジ角度を入手
        angle = np.arctan2(fy, fx)

        return edge, angle
    
    def angle_quantization(self, angle):
        angle = angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5)&(angle <= 67.5))] = 45
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
                if angle[y, x] == 45:
                    dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                if angle[y, x] == 90:
                    dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                if angle[y, x] == 135:
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
                
                if max(max(edge[y, x], edge[y+dy1, x+dx1]), edge[y+dy2, x+dx2]) != edge[y, x]:
                    _edge[y,x] = 0
                
        return _edge
    
    def hysterisis(self, edge, HT=100, LT=30):
        H, W = edge.shape

        # ヒステリシスのしきい値
        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        _edge = np.zeros((H+2, W+2), dtype=np.float32)
        _edge[1:H+1, 1:W+1] = edge

        # Nearest neighbor(最近傍法)
        nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

        for y in range(1, H+2):
            for x in range(1, W+2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0
                
        edge = _edge[1:H+1, 1:W+1]

        return edge
    
    def all_process_run(self):
        # グレースケール処理
        gray = self.BGR2GRAY(self.img)
        # ガウシアンフィルタ
        gaussian = self.gaussian_filter(gray, K_size=5, sigma=1.4)
        # ソーベルフィルタ
        fy, fx = self.sobel_filter(gaussian, K_size=3)
        # エッジ強度と角度の取得
        edge, angle = self.get_edge_angle(fx, fy)
        # 角度の量子化
        angle = self.angle_quantization(angle)
        # 非最大値以外の除去
        edge = self.non_maximum_suppression(angle, edge)
        # ヒステリシス変換
        out = self.hysterisis(edge, 100, 30)
        self.result = out.copy()
        return out



    
class Hough_Line_process:
    def __init__(self, img):
        self.img = img
    
    def voting(self, edge):
        H, W = edge.shape
        drho = 1
        dtheta = 1

        # 画像における対角線の長さ(直線における最大長)を取得
        rho_max = np.ceil(np.sqrt(H**2 + W**2)).astype(np.int16)

        # hough テーブル
        # 2次元配列を作成する
        # 行(rho_max*2)は直線の位置
        # 列(180)は直線の角度(0 ≦ Θ ≦ 180°)になる
        hough = np.zeros((rho_max * 2, 180), dtype=np.int16)

        # エッジのインデックスを取得する
        ind = np.where(edge == 255)

        ## hough 変換
        # 画素値が255の画素の数だけ繰り返す
        for y, x in zip(ind[0], ind[1]):
            # 1画素に対して0から180までの180通りを繰り返す
            for theta in range(0, 180, dtheta):
                # 度数法からラジアンへ変換する
                t = np.pi / 180 * theta

                # Hough変換の式
                rho = int(x*np.cos(t) + y * np.sin(t))

                # ここでrho + rho_maxなのはrhoが負の値をとる可能性があるため
                # rho = -10, rho_max = 240の時230になる
                # cosとsinは-1から1までとる
                # つまり基準点からの直線距離(xとyの比率は変わらない)に対して収縮と拡張により
                # 同一直線状の場合はhoughテーブルの同じ位置に投票することになり一定のsin波とcos波を形成することになる
                hough[rho + rho_max, theta] += 1
        
        out = hough.astype(np.uint16)

        return out
    
    def all_process_run(self):
        out = self.voting(self.img)

        return out.astype(np.uint8)

img_file_path = "../thorino.jpg"
process_1 = Canny_process(img_file_path)
process_1_img = process_1.all_process_run()
process_2 = Hough_Line_process(process_1_img)
result = process_2.all_process_run()
cv2.imwrite("./result/Question_44.jpg", result)

