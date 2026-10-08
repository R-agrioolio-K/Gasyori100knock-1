# 平均値フィルタ
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.mean_img = None

    def mean_filter(self, kernel_size=3):
        if len(self.img.shape) == 3:
            image = self.img.copy()
            H, W, C = image.shape
        else:
            image = np.expand_dims(self.img.copy(), -1)
            H, W, C = image.shape
        
        padding_size = kernel_size // 2
        out = np.zeros(((H + padding_size*2), (W + padding_size*2), C), np.float32)
        out[padding_size : H + padding_size, padding_size : W + padding_size] = image.copy()

        # フィルタ処理の開始
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[padding_size + y, padding_size + x, c] = np.mean(out[y : y + kernel_size, x : x + kernel_size, c])
        
        self.mean_img = out[padding_size : H + padding_size, padding_size : W + padding_size].astype(np.uint8)


img_file_path = "../imori.jpg"
process = process_class(img_file_path)

process.mean_filter()

cv2.imwrite("./result/Question_11.jpg", process.mean_img)
#cv2.show("mean filter image", process.mean_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()