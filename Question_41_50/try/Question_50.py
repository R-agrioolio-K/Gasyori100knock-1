# クロージング処理
# クロージング処理ではモルフォロジー処理の膨張をN回行った後に収縮をN会行う処理
# 最初にキャニーエッジ検出を行う
import cv2
import numpy as np

class canny_class:
    def __init__(self, img:np.array)-> np.array:
        self.img = img.copy()

    # グレースケール変換
    def BGR2GRAY(self, img)-> np.array:
        b = img[:,:, 0].copy()
        g = img[:,:, 1].copy()
        r = img[:,:, 2].copy()

        gray = 0.2126*r + 0.7152*g + 0.0722*b
        gray = gray.astype(np.uint8)

        return gray

    # ガウシアンフィルタ
    def gaussian_filter(self, img, K_size=3, sigma=1.3)-> np.array:
        # グレースケール画像でない場合でも処理できるように分岐を行う
        if len(img.shape) == 3:
            gray = False
            H, W, C = img.shape
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
            gray = True
        
        # パディング処理
        # 整数除算
        pad = K_size // 2
        # 出力用配列の作成
        out = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float16)
        out[pad:H+pad, pad:W+pad] = img.copy()

        # フィルタの定義
        K = np.zeros((K_size, K_size), dtype=np.float16)
        for x in range(-pad, -pad+K_size):
            for y in range(-pad, -pad+K_size):
                K[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (2*sigma*sigma))
        
        K /= (2*np.pi*sigma*sigma)
        K /= K.sum()

        # 出力用画像を一時保存配列へコピー
        tmp = out.copy()

        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[y+pad, x+pad, c] = np.sum(K*(tmp[y:y+K_size, x:x+K_size, c]))
        
        out = out[pad:H+pad, pad:W+pad]

        if gray:
            out = out[:,:, 0]
        
        return out
    
    # ソーベルフィルタ
    def sobel_filter(self, img, K_size=3)-> tuple[np.array, np.array]:
        # 画像の縦横サイズを取得
        H, W = img.shape

        # ゼロパディング
        # 整数除算
        pad = K_size // 2
        # 出力用配列を作成
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float16)
        out[pad:H+pad, pad:W+pad] = img.copy()

        # フィルタの作成
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

        # 縦エッジ、横エッジそれぞれの出力用配列の定義
        out_v = out.copy()
        out_h = out.copy()

        for y in range(H):
            for x in range(W):
                out_v[y+pad, x+pad] = np.sum(Kv*out[y:y+K_size, x:x+K_size])
                out_h[y+pad, x+pad] = np.sum(Kh*out[y:y+K_size, x:x+K_size])
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad:H+pad, pad:W+pad]
        out_h = out_h[pad:H+pad, pad:W+pad]

        return out_v, out_h


    # エッジ強度と角度の取得
    def get_edge_angle(self, fx, fy)-> tuple[np.array, np.array]:
        # エッジ強度の取得
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        # エッジ角度の取得
        # 初めにゼロ除算を防ぐため配列の値の型を少数に変換
        fx = fx.astype(np.float32)
        fy = fy.astype(np.float32)

        # ゼロ除算を防ぐため0の数値を置き換え
        fx[np.where(fx == 0)] = 1e-10
        #print("np.where(fx == 0)", np.where(fx==0))

        angle = np.arctan(fy / fx)

        return edge, angle
    
    # 角度の量子化
    def angle_quantization(self, angle)-> np.array:
        # ラジアンから角度法への変換
        angle = (angle/np.pi) * 180
        # 角度の量子化
        # -22.5よりも小さい値を正の角度に変換する
        angle[angle<-22.5] = 180 + angle[angle < -22.5]
        
        # 量子化角度における保存用配列の作成
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

        return _angle
    
    # 細線化(非極大値のの抑制)
    def non_maximum_suppression(self, angle, edge)-> np.array:
        # 画像サイズの取得
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
                # 画像からのはみ出しに対する処理
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
                
                # エッジ強度の確認と変換
                # 隣り合うがその最大値が注目画素と異なるのならば注目画素を0にする
                # 隣り合う画素というのはその画素のエッジ角度の垂直方向である
                if max(max(edge[y, x], edge[y+dy1, x+dx1]), edge[y+dy2, x+dx2]) != edge[y, x]:
                    _edge[y, x] = 0
        
        return _edge        
    
    # ヒステリシス変換
    def hysterisis(self, edge, HT=100, LT=30)-> np.array:
        # 画像サイズを取得
        H, W = edge.shape

        # ヒステリシスしきい値による変換
        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        # 出力用画像の配列を作成する
        _edge = np.zeros((H+2, W+2), dtype=np.float16)
        _edge[1:H+1, 1:W+1] = edge

        # 最近傍補完
        # フィルタの作成
        nn = np.array((
            (1., 1., 1.),
            (1., 0., 1.),
            (1., 1., 1.)
        ), dtype=np.float16)

        # 処理の開始
        for y in range(1,H+2):
            for x in range(1,W+2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                if np.max(_edge[y-1:y+2, x-1:x+2] ** nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0
        
        edge = _edge[1:H+1, 1:W+1]

        return edge
    
    def all_run(self):
        # グレースケール変換
        gray = self.BGR2GRAY(self.img)
        # ガウシアンフィルタ
        out = self.gaussian_filter(gray, K_size=5, sigma=1.4)
        # ソーベルフィルタ
        fy, fx = self.sobel_filter(out, K_size=3)
        # エッジ強度と角度の取得
        edge, angle = self.get_edge_angle(fx, fy)
        # エッジ角度量子化
        angle = self.angle_quantization(angle)
        # 細線化(非極大値抑制)
        edge = self.non_maximum_suppression(angle, edge)
        # ヒステリシス閾値処理
        out = self.hysterisis(edge, 50, 20)

        return out

class Morphology_class:
    def __init__(self, img:np.array):
        self.img = img.copy()
    
    # モルフォロジー変換(収縮)
    def Morphology_Erode(self, img, Erode_time=1):
        H, W = img.shape
        out = img.copy()

        # フィルタの作成
        MF = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]

        # 処理の開始
        for i in range(Erode_time):
            tmp = np.pad(out, (1, 1), 'edge')
            for y in range(1, H+1):
                for x in range(1, W+1):
                    if np.sum(MF*tmp[y-1:y+2, x-1:x+2]) < 255*4:
                        out[y-1, x-1] = 0
        
        return out
    
    # モルフォロジー変換(拡大)
    def Morphology_Dilate(self, img, Dil_time=1):
        H, W = img.shape

        MF = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]

        #処理の開始
        out = img.copy()
        for i in range(Dil_time):
            tmp = np.pad(out, (1, 1), 'edge')
            for y in range(1, H+1):
                for x in range(1, W+1):
                    if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                        out[y-1, x-1] = 255
        
        return out
    
    def Morphology_Closing(self, time=1):
        out = self.Morphology_Dilate(self.img, Dil_time=time)
        out = self.Morphology_Erode(out, Erode_time=time)

        return out.astype(np.uint8)
    
img_file_path = "../imori.jpg"
img = cv2.imread(img_file_path)

# キャニーエッジ検出
canny_ins = canny_class(img)
out = canny_ins.all_run()

morphology_ins = Morphology_class(out)
result = morphology_ins.Morphology_Closing()

cv2.imwrite("./result/Question_50.jpg", result)


