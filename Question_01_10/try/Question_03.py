# 二値化処理に関するプログラム
import cv2
import numpy as np
class img_Gray_Binarization:
# グレースケール変換
    def __init__(self, img):
        self.img = img

    def BGR2GRAY(self):
        img = self.img

        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()
        
        # グレースケールへの変換式
        gray = 0.2126 * r + 0.7125 * g + 0.0722 * b
        out = gray.astype(np.uint8) #gray配列をint型に変換する
        self.gray = out


    # 二値化処理
    def binarization(self, th=128):
        # 引数thは二値化するためのしきい値を示している
        if self.gray is not None:
            img = self.gray
        else:
            img = self.BGR2GRAY()

        img[img < th] = 0
        img[img >= th] = 255
        
        self.binarized = img

# 画像の読み込み
img_file_path = "../imori.jpg"
img = cv2.imread(img_file_path)

# インスタンス化
img_processor = img_Gray_Binarization(img)

# グレースケール変換の実行
img_processor.BGR2GRAY()

# 二値化処理の実行
img_processor.binarization(th = 128)

# 二値化画像の取り出し
binarized_img = img_processor.binarized

# 二値化画像の保存
cv2.imwrite("./result/Question_03.jpg", binarized_img)
#cv2.imshow("Question_03_result", binarized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


