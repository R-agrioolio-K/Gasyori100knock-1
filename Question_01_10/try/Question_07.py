# 平均プーリングを行うプログラム
import cv2
import numpy as np

class img_process():
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.pool_img = None
        self.part_of_img = None
    
    def average_pool(self, G=8):
        out = self.img.copy()

        H, W, C = out.shape
        Nh = int(H / G)
        Nw = int(W / G)

        for y in range(Nh):
            for x in range(Nw):
                if self.part_of_img is None:
                    # 画像の一部分(G, G, チャンネル数)を切り抜くことができるかを試した
                    self.part_of_img = out[G*y:G*(y+1), G*x:G*(x+1), 0]
                    #print("part_of_img : ",self.part_of_img)
                for c in range(C):
                    out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int8)

        self.pool_img = out

img_file_path = "../imori.jpg"

process_ins = img_process(img_file_path)

process_ins.average_pool()

cv2.imwrite("./result/Question_07.jpg", process_ins.pool_img)
#cv2.imshow("pooling_image", process_ins.pool_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# チャンネル数が1つの時は自動的にグレースケール画像に変換され保存される
cv2.imwrite("./result/Question_07_part.jpg", process_ins.part_of_img)
#cv2.imshow("pooling_image", process_ins.pool_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
        