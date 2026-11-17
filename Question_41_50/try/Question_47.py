# モルフォロジー処理
# 大津の二値化したものにモルフォロジー処理による膨張を2回行う
import cv2
import numpy as np

class molforogy_process:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
    
    # グレースケール変換
    def BGR2GRAY(self, img) -> np.array:
        b = img[:,:, 0].copy()
        g = img[:,:, 1].copy()
        r = img[:,:, 2].copy()

        #グレースケール変換
        gray = 0.2126*r + 0.7152*g + 0.0722*b
        gray = gray.astype(np.uint8)

        return gray
    
    # 大津の二値化を行うメソッド
    def otsu_binarization(self, img, th=128) -> np.array:
        # 画像のサイズを取得する
        H, W = img.shape
        out = img.copy()

        max_sigma = 0
        max_t = 0

        # 閾値を決定する処理
        for _t in range(1, 255):
            # 現在のしきい値 _t　よりも小さい画素値のインデックス番号を取得
            v0 = out[np.where(out < _t)]
            # 閾値 _t よりも小さい画素値の平均値を取得
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            # 合計画素数でしきい値_tよりも小さい画素数を割る
            w0 = len(v0) / (H * W)
            # _tよりも大きいものに同様の処理
            v1 = out[np.where(out >= _t)]
            m1 = np.mean(v1)
            w1 = len(v1) / (H*W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)
            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t
        
        # 二値化
        print("threshold >>", max_t)
        th = max_t
        out[out < th] = 0
        out[out >= th] = 255

        return out
    
    def Morphology_Erode(self, img, Dil_time=1):
        # 画像のサイズを取得
        H, W = img.shape

        # フィルタを設定
        MF = np.array((
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0)
        ), dtype=np.uint8)

        # モルフォロジー変換
        out = img.copy()
        # Dil_timeはモルフォロジー変換の実行回数
        for i in range(Dil_time):
            tmp = np.pad(out, (1, 1), 'edge')
            for y in range(1, H+1):
                for x in range(1, W+1):
                    if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                        out[y-1, x-1] = 255
        
        return out
    
    def all_run(self):
        gray = self.BGR2GRAY(self.img)
        otsu = self.otsu_binarization(gray)
        out = self.Morphology_Erode(otsu, Dil_time=2)

        return out

img_file_path = "../imori.jpg"
molforogy_ins = molforogy_process(img_file_path)
result = molforogy_ins.all_run()
cv2.imwrite("./result/Question_47.jpg", result)
        

