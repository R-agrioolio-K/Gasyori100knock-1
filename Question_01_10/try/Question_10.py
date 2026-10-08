# メディアンフィルタによる平滑化を行う
# フィルタサイズ3×3を実装する
import cv2
import numpy as np

class median_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.median_img = None
    
    def median_filter(self, kernel_size = 3):
        if len(self.img.shape) == 3:
            image = self.img.copy()
            H, W, C = image.shape
        else:
            image = np.expand_dims(self.img.copy(), -1)
            H, W, C = image.shape
        
        # ゼロパディング
        pad = kernel_size // 2
        out = np.zeros(((H + pad*2), (W + pad*2), C), dtype=np.float32)
        out[pad:pad+H, pad:pad+W] = image.copy().astype(np.float32)

        tmp = out.copy()

        #フィルタリング処理の開始
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad+y, pad+x, c] = np.median(tmp[y:y+kernel_size, x:x+kernel_size, c])

        out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

        self.median_img = out

img_file_path = "../imori_noise.jpg"

process = median_class(img_file_path)
process.median_filter()

cv2.imwrite("./result/Question_10.jpg", process.median_img)
#cv2.imshow("result median image", process.median_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
        
