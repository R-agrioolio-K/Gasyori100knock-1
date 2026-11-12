# Cannyエッジ検出を3Stepに分ける
# エッジ強度について実装する
"""
Canny法は
1. ガウシアンフィルタを掛ける
2. x, y方向のSobelフィルタを掛ける。
3. エッジ勾配の値からNon-maximum suppressionによりエッジの細線化を行う
4. ヒストリシスによる敷居値処理を行う
"""
import cv2
import numpy as np

class canny_step1:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.tmp = None
        self.sobel_v = None
        self.sobel_h = None
        self.edge = None
        self.angle = None
    
    
    def all_process_run(self):
        self.BGR2GRAY()
        self.gaussian_filter()
        self.sobel_filter()
        self.get_edge_angle()
        self.angle_quantization()
        return self.edge.astype(np.uint8), self.angle.astype(np.uint8)
    
    def BGR2GRAY(self):
        b = self.img[:,:, 0]
        g = self.img[:,:, 1]
        r = self.img[:,:, 2]

        out = 0.2126*r + 0.7152*g + 0.0722*b
        out = out.astype(np.uint8)

        self.tmp = out
    
    # ガウシアンフィルタ
    def gaussian_filter(self, K_size=3, sigma=1.3):
        if len(self.tmp.shape) == 3:
            H, W, C = self.tmp.shape
            tmp = self.tmp.copy()
            gray = False
        else:
            tmp = np.expand_dims(self.tmp, axis=-1)
            H, W, C = tmp.shape
            gray = True
        
        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H + pad*2, W+pad*2, C), dtype=np.float32)
        out[pad: pad+H, pad: pad+W] = tmp.astype(np.float32)

        # カーネルの作成をする
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for x in range(-pad, -pad+K_size):
            for y in range(-pad, -pad+K_size):
                K[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (2*(sigma**2)))
        
        K /= (2*np.pi*sigma*sigma)
        K /= K.sum()

        tmp = out.copy()

        # フィルタ処理
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad+y, pad+x, c] = np.sum(K*tmp[y:y+K_size, x:x+K_size, c])
        
        out = np.clip(out, 0, 255)
        out = out[pad: pad+H, pad: pad+W]
        out = out.astype(np.uint8)

        if gray:
            out = out[..., 0]

        self.tmp = out
    
    # ソーベルフィルタ
    def sobel_filter(self, K_size=3):
        if len(self.tmp.shape) == 3:
            H, W, C = self.tmp.shape
            tmp = self.tmp.copy()
        else:
            tmp = self.tmp.copy()
            H, W = tmp.shape
        
        # ゼロパディング
        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad: pad+H, pad: pad+W] = tmp.astype(np.float32)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        # ソーベルフィルタを作成する
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
                out_v[pad+y, pad+x] = np.sum(Kv * (tmp[y:y+K_size, x:x+K_size]))
                out_h[pad+y, pad+x] = np.sum(Kh * (tmp[y:y+K_size, x:x+K_size]))
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
        out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)

        self.sobel_v = out_v
        self.sobel_h = out_h
    
    def get_edge_angle(self):
        # エッジ強度を取得
        fx = self.sobel_h.copy()
        fy = self.sobel_v.copy()
        # np.powerは配列の要素ごとに累乗の計算を行う
        # np.powerは配列でなくても構わない
        edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
        # 配列fxの値が0の時、1^-5に置き換える
        # ゼロ徐算を防ぐ
        #fx = np.maximum(fx, 1e-5)

        # エッジ角度を取得
        #arctan2は，y/xの計算から角度を求めるメソッド
        angle = np.arctan2(fy , fx)

        self.edge = edge
        self.angle = angle
    
    def angle_quantization(self):
        # 角度をradianからdegreeに直す
        angle = self.angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

        self.angle = _angle
    

img_file_path = "../imori.jpg"
processor = canny_step1(img_file_path)
edge, angle = processor.all_process_run()
cv2.imwrite("./result/Question_41_1.jpg", edge)
cv2.imwrite("./result/Question_41_2.jpg", angle)





    

