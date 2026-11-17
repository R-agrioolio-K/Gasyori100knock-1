# モルフォロジー勾配
# 大津の二値化を行った後に、モルフォロジー勾配を求める
# モルフォロジー勾配とはモルフォロジー膨張の画像と収縮の画像の差分をとることで物体の境界線を抽出する
import cv2
import numpy as np

# モルフォロジー勾配を求めるクラス
class process_1:
    def __init__(self, img):
        self.img = img
        
    
    def BGR2GRAY(self, img):
        b = img[:,:, 0].copy()
        g = img[:,:, 1].copy()
        r = img[:,:, 2].copy()

        gray = r*0.2126 + g*0.7152 + b*0.0722
        gray = gray.astype(np.uint8)

        return gray
    
    def otsu_binarization(self, img, th=128):
        H, W = img.shape
        out = img.copy()

        max_sigma = 0
        max_t = 0

        # 閾値を決定する
        for _t in range(1, 255):
            # 仮のしきい値よりも小さい画素に対するパラメータ計算
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H*W)
            # 仮のしきい値よりも大きい画素に対するパラメータ計算
            v1 = out[np.where(out > _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H*W)
            # シグマを計算する
            sigma = w0 * w1 * ((m0 - m1)**2)
            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t
            
        # 二値化処理
        print("threshold = ",max_t)
        th = max_t
        out[out < th] = 0
        out[out >= th] = 255

        return out
    
    # モルフォロジー変換による拡大
    def Erode(self, img, Erode_time=1):
        H, W = img.shape
        out = img.copy()

        MF = np.array((
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0)
        ), dtype=np.uint8)

        # 処理開始
        for i in range(Erode_time):
            # パディングによる一時保存の配列作成
            # 処理たびに代わるため処理回数ループの中に作成
            # 全方向に対して1次元追加(上下左右に対して)
            tmp = np.pad(out, pad_width=1, mode='edge')
            # tmp.shape = (H+2, W+2)実際の配列では0からH+1まで
            for y in range(1,H+1):
                for x in range(1,W+1):
                    if np.sum(MF*tmp[y-1:y+2, x-1:x+2]) < 255*4:
                        out[y-1, x-1] = 0
        
        return out
    
    # モルフォロジー変換における拡大処理
    def Dilate(self, img, Dil_time=1):
        H, W = img.shape
        out = img.copy()

        # フィルタの作成
        MF = np.array((
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0)
        ), dtype=np.uint8)

        # 実際の処理開始
        for i in range(Dil_time):
            tmp = np.pad(out, pad_width=1, mode='edge')
            for y in range(1,H+1):
                for x in range(1,W+1):
                    if np.sum(MF*tmp[y-1:y+2, x-1:x+2]) >= 255:
                        out[y-1,x-1] = 255
        
        return out
    
    # グレースケール化 -> 大津の二値化 -> モルフォロジー縮小を一括で行う
    def run_Erode_1(self):
        # グレースケール化
        gray = self.BGR2GRAY(self.img)
        # 大津の二値化
        otsu = self.otsu_binarization(gray)
        # モルフォロジー縮小
        eroded = self.Erode(otsu)

        return eroded
    
    # グレースケール化 -> 大津の二値化 -> モルフォロジー拡大を一括で行う
    def run_Dilate_1(self):
        # グレースケール化
        gray = self.BGR2GRAY(self.img)
        # 大津の二値化
        otsu = self.otsu_binarization(gray)
        # モルフォロジー拡大
        dilated = self.Dilate(otsu)

        return dilated

# 画像のパスの設定
img_file_path = "../imori.jpg"
# 画像の読み込み
img = cv2.imread(img_file_path)

# クラスのインスタンス化
process_ins = process_1(img)
# モルフォロジーによる縮小処理
eroded = process_ins.run_Erode_1()
# モルフォロジーによる拡大処理
dilated = process_ins.run_Dilate_1()

# 差分をとってその絶対値に対して255倍にする
# モルフォロジーでは画素値が0か255のはずなのになぜ255倍?
out = np.abs(eroded - dilated) * 255
cv2.imwrite("./result/Question_51.jpg", out)


    
