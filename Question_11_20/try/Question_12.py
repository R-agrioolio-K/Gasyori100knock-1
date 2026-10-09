# モーションフィルタによる処理を行う
# モーションフィルタは体格方向の平均値をとるフィルタ
"""
モーションフィルタの形状
[
[1/3, 0, 0],
[0, 1/3, 0],
[0, 0, 1/3]
]

"""
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.motion_img = None
    
    def motion_filter(self, K_size=3):
        filter = np.array([
            [[1/3, 0, 0],
            [0, 1/3, 0],
            [0, 0, 1/3]],
            [[1/3, 0, 0],
            [0, 1/3, 0],
            [0, 0, 1/3]],
            [[1/3, 0, 0],
            [0, 1/3, 0],
            [0, 0, 1/3]]
            ])
        if len(self.img.shape) == 3:
            image = self.img.copy()
            H, W, C = image.shape
        else:
            image = np.expand_dims(self.img.copy(), -1)
            H, W, C = image.shape
        
        pad = K_size // 2
        out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float32)
        out[pad : H + pad, pad : W + pad] = image.copy().astype(np.float32)
        tmp = out.copy()

        # フィルタ処理
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(filter[0:3, 0:3, c] * tmp[y : y + K_size, x : x + K_size, c])

        out = out[pad : pad + H, pad : pad + W].astype(np.uint8)

        self.motion_img = out
    

img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.motion_filter()

cv2.imwrite("./result/Question_12.jpg", process.motion_img)
#cv2.imshow("motion filter image", process.motion_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


