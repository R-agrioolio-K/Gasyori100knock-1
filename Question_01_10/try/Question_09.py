# ガウシアンフィルタを実装するためのプログラム
import cv2
import numpy as np

class all_process:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.gaus_image = None

    def Gaus_filter(self, K_size=3, sigma=1.3):
        img = self.img.copy()
        # 画像の次元数が3よりも小さいかの判定
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            # axis -1は何?
            img = np.expand_dism(img, axis=-1)
            H, E, C = out.shape
        
        # ゼロパディングを行う
        # フィルタは奇数しかありえないから2で割って切り捨てた数だけサイズを追加する
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float64)
        out[pad : pad + H, pad: pad+W] = img.copy().astype(np.float64)
        

        K = np.zeros((K_size, K_size), dtype=np.float64)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x + pad] = np.exp(-(x**2 + y**2) / (2*(sigma**2)))
        K /= (2 * np.pi * sigma * sigma)
        K /= K.sum()

        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
        # clipは何?
        out = np.clip(out, 0, 255)
        out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

        self.gaus_image = out

img_file_path = "../imori_noise.jpg"

process = all_process(img_file_path)

process.Gaus_filter()

cv2.imwrite("./result/Question_09.jpg", process.gaus_image)
#cv2.imshow("gause_result image", process.gause_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


