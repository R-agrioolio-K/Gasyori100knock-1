# 減色処理(ポスタリゼーション)を行うためのプログラム
import cv2
import numpy as np

class img_process():
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.decrease_img = None
    
    def decrease_color(self):
        out = self.img.copy()

        # 演算子//は切り捨て徐算
        out = ((out // 64) * 64) + 32

        self.decrease_img = out
    

img_file_path = "../imori.jpg"

process = img_process(img_file_path)

process.decrease_color()

cv2.imwrite("./result/Question_06.jpg", process.decrease_img)
#cv2.imshow("decrease_img", process.decrease_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
