# Cannyエッジ検出におけるヒステリシス閾値処理
# ヒストリシス閾処理では勾配強度の二値化を行う
# 閾値は2つの(HT: high threshhold, LT: low threshold)を用いる
import cv2
import numpy as np

class canny_step3:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)

    def BGR2GRY(self):
        b = self.img[:,:, 0]
        g = self.img[:,:, 1]
        r = self.img[:,:, 2]

        gray = 0.2126*r + 0.7152*g + 0.0722*b
        gray = gray.astype(np.uint8)

        return gray
    
    # カラー画像に対して処理が可能なようにグレースケール画像の場合もチャネル方向の次元を1にする
    def gaussian_filter(self, img, K_size=3, sigma=1.3):
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
        
        # ゼロパディングの処理
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float32)
        out[pad:H+pad, pad:W+pad] = img.copy().astype(np.float32)

        # フィルタのパラメータ準備
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for x in range(-pad, -pad+K_size):
            for y in range(-pad, -pad+K_size):
                K[y+pad, x+pad] = np.exp(-(x**2 + y**2)/(2*(sigma**2)))
        
        K /= (2*np.pi*sigma*sigma)
        K /= K.sum()

        tmp = out.copy()

        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad+y, pad+x, c] = np.sum(K*tmp[y:y+K_size, x:x+K_size, c])
        
        out = np.clip(out, 0, 255)
        out = out[pad:pad+H, pad:pad+W]
        out = out.astype(np.uint8)
        out = out[..., 0]

        return out
    
    def sobel_filter(self, img, K_size=3):
        # ゼロパディング
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            H, W = img.shape
        
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float32)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        # ソーベルフィルタの定義
        Kv = [
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]
        ]

        Kh = [
            [1., 0., -1],
            [2., 0., -2],
            [1., 0., -1]
        ]

        for y in range(H):
            for x in range(W):
                out_v[pad+y, pad+x] = np.sum(Kv*(tmp[y:y+K_size, x:x+K_size]))
                out_h[pad+y, pad+x] = np.sum(Kh*(tmp[y:y+K_size, x:x+K_size]))
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
        out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)

        return out_v, out_h
    
    def get_edge_angle(self, fx, fy):
        # エッジ強度を取得する
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        edge = np.clip(edge, 0, 255)

        angle = np.arctan2(fy, fx)

        return edge, angle
    
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
                    dx1, dy1, dx2, dy2 = 0, -1, 0, 1
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
                
                if max(max(edge[y, x], edge[y+dy1, x+dx1]), edge[y+dy2, x+dx2]) != edge[y, x]:
                    _edge[y, x] = 0
        return _edge
    
    def hysterisis(self, edge, HT=100, LT=30):
        H, W = edge.shape

        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        _edge = np.zeros((H+2, W+2), dtype=np.float32)
        _edge[1:H+1, 1:W+1] = edge

        nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

        for y in range(1, H+2):
            for x in range(1, W+2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                if np.max(_edge[y-1:y+2, x-1:x+2]**nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0
        
        edge = _edge[1:H+1, 1:W+1]

        return edge
    
    def all_process_run(self):
        gray = self.BGR2GRY()
        gaussian = self.gaussian_filter(gray, K_size=5, sigma=1.4)
        fy, fx = self.sobel_filter(gaussian, K_size=3)
        edge, angle = self.get_edge_angle(fx, fy)
        angle = self.angle_quantization(angle)
        edge = self.non_maximum_suppression(angle, edge)
        out = self.hysterisis(edge, 50, 20)

        return out.astype(np.uint8)
    


img_file_path = "../imori.jpg"
processor = canny_step3(img_file_path)
out = processor.all_process_run()

cv2.imwrite("./result/Question_43.jpg", out)

