# 大津の二値化をするためのプログラム
import cv2
import numpy as np

class img_processor:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path).astype(np.float32)
        self.gray = None
        self.otsu_binarized = None
    
    # グレースケール変換
    def BGR2GRAY(self):
        b = self.img[:, :, 0].copy()
        g = self.img[:, :, 1].copy()
        r = self.img[:, :, 2].copy()

        # グレースケール変換の式
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
        self.gray = gray.astype(np.uint8)
    
    # 大津の二値化
    def otsu_binarization(self):
        if self.gray is None:
            self.BGR2GRAY()
        
        gray = self.gray.copy()

        max_sigma = 0
        max_t = 0
        H, W = gray.shape
        # しきい値を決定する
        for _t in range(1, 256):
            # _tをしきい値としてそれ以下の画素値をすべてリストに格納する
            # v0は1次元配列
            v0 = gray[np.where(gray < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            # 輝度値の低い画素が全体に対して占める割合を計算する
            w0 = len(v0) / (H * W)


            v1 = gray[np.where(gray >= _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H * W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)
            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t
        
        # 二値化
        print("threshold >>", max_t)
        th = max_t
        gray[gray < th] = 0
        gray[gray >= th] = 255
        self.otsu_binarized = gray

# 画像パスの定義
img_file_path = "../imori.jpg"

# インスタンス化
img_trans_instance = img_processor(img_file_path)

# グレースケール化
img_trans_instance.BGR2GRAY()
# 大津の二値化(部分判別法)
img_trans_instance.otsu_binarization()

# 二値化画像の保存
cv2.imwrite("./result/Question_04.jpg", img_trans_instance.otsu_binarized)
#cv2.imshow("Question_04_result", img_trans_instance.otsu_binarized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
