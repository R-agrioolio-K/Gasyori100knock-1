# ガンマ補正を実行するプログラムファイル
# ガンマ補正とは、カメラなどの媒体の経由によって画素値が非線形的に変換された場合の補正
import cv2
import numpy as np
import matplotlib.pyplot as plt

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gamma = None
    
    def gamma_correction(self, c=1, g=2.2):
        out = self.img.copy().astype(np.float32)
        out /= 255. # 画素値を0~1に正規化
        out = ((1/c) * out) ** (1/g)
        out *= 255
        self.gamma = out.astype(np.uint8)

img_file_path = "../imori_gamma.jpg"

process = process_class(img_file_path)
process.gamma_correction()

cv2.imwrite("./result/Question_24.jpg",process.gamma)
#cv2.imshow("result gamma",process.gamma)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

