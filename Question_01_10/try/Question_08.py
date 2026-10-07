# Maxプーリングを行うプログラム
import cv2
import numpy as np

class img_process():
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.max_pool_img = None
    
    def max_pool(self, G=8):
        out = self.img.copy()

        # 画像形状の取得
        H,W,C = out.shape

        # グリッドと画像形状から縦横におけるループ回数を算出
        Gh = int(H / G)
        Gw = int(W / G)
        
        for y in range(Gh):
            for x in range(Gw):
                for c in range(C):
                    out[G*y:G*(y+1), G*x:G*(x+1), c] = np.max(out[G*y:G*(y+1), G*x:G*(x+1), c])

        
        self.max_pool_img = out


# 画像ファイル名の定義
img_file_path = "../imori.jpg"

process = img_process(img_file_path)

process.max_pool()

# 画像の保存
cv2.imwrite("./result/Question_08.jpg", process.max_pool_img)
#cv2.imshow("max_pool image", process.max_pool_img)
#cv2.waitKey(0)
#cv2.destroyAllWindow()