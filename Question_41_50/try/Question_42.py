# Cannyエッジ検出における細線化を実装するプログラム
import cv2
import numpy as np
import matplotlib.pyplot as plt

class canny_step2:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
    
    def all_process_run(self):
        gray = self.BGR2GRAY()
        gaussian = self.gaussian_filter(gray, K_size=5, sigma=1.4)
        fy, fx = self.sobel_filter(gaussian, K_size=3)
        edge, angle = self.get_edge_angle(fx, fy)
        angle = self.angle_quantization(angle)
        edge = self.non_maximum_suppression(angle, edge)
        return edge, angle

    def BGR2GRAY(self):
        img = self.img
        b = img[:,:, 0].copy()
        g = img[:,:, 1].copy()
        r = img[:,:, 2].copy()

        out = 0.2126*r + 0.7152*g + 0.0722*b
        out = out.astype(np.uint8)

        return out
    
    # ガウシアンフィルタを実行する関数
    def gaussian_filter(self, img, K_size=3, sigma=1.3):
        # 画像がカラー画像、グレースケール画像のどちらかを判定する
        # グレースケール画像の場合は、チャンネル数を1にする
        # カラー画像でも処理できるようにするためにグレースケールがぞうの場合もチャンネル次元を増やしている
        if len(img.shape) == 3:
            H, W, C = img.shape
            tmp = img.copy()
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
        
        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float32)

        # カーネルの準備をする
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for x in range(-pad, -pad+K_size):
            for y in range(-pad, -pad+K_size):
                K[y+pad, x+pad] = np.exp(- (x**2 + y**2)/ (2*(sigma**2)))
        
        K /= (2*np.pi*sigma*sigma)
        K /= K.sum()

        tmp = out.copy()

        # 実際のフィルタ処理を行う
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
        if len(img.shape) == 3:
            H, E, C = img.shape
        else:
            H, W = img.shape
        
        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float32)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        # ソーベルフィルタを定義する
        Kv = [
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1]
        ]
        Kh = [
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
        ]

        # フィルタリング処理
        for y in range(H):
            for x in range(W):
                out_v[pad+y, pad+x] = np.sum(Kv * (tmp[y:y+K_size, x:x+K_size]))
                out_h[pad+y, pad+x] = np.sum(Kh * (tmp[y:y+K_size, x:x+K_size]))
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
        out_h = out_h[pad:pad+W, pad:pad+W].astype(np.uint8)

        return out_v, out_h
    
    # 縦横のエッジを合力として表現しその角度を表現するメソッド
    def get_edge_angle(self, fx, fy):
        edge = np.sqrt(np.power(fx.astype(np.float32), 2)+np.power(fy.astype(np.float32), 2))
        edge = np.clip(edge, 0, 255)

        angle = np.arctan2(fy, fx)

        return edge, angle
    
    def angle_quantization(self, angle):
        angle = angle / np.pi*180
        angle[angle < -22.5] = 180+angle[angle < -22.5]
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


img_file_path = "../imori.jpg"
processor = canny_step2(img_file_path)
edge, angle = processor.all_process_run()
edge = edge.astype(np.uint8)
angle = angle.astype(np.uint8)

cv2.imwrite("./result/Question_42_1.jpg", edge)
cv2.imwrite("./result/Question_42_2.jpg", angle)

