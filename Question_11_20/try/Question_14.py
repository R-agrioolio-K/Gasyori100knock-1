# 微分フィルタを実装する
import cv2
import numpy as np

# 処理クラスの定義
class process_class:
    def __init__(self, img_file_path, K_size):
        self.img = cv2.imread(img_file_path)
        self.K_size = K_size
        self.gray_img = None
        self.differential_vimg = None
        self.differential_himg = None
    
    def BGR2GRAY(self):
        b = self.img[:,:, 0]
        g = self.img[:,:, 1]
        r = self.img[:,:, 2]

        # グレースケール変換
        gray = 0.2126*r + 0.7152*g + 0.0722*b
        self.gray_img = gray.astype(np.uint8)
    
    def different_filter(self):
        if len(self.gray_img.shape) == 3:
            H, W, C = self.gray_img.shape
            img = self.gray_img.copy().astype(np.float32)
        else:
            img = np.expand_dims(self.gray_img, axis=-1)
            H, W, C = img.shape
        
        # ゼロパディング
        pad = self.K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float32)
        out[pad:H+pad, pad:W+pad] = self.gray_img.copy().astype(np.float32)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        # 垂直方向微分フィルタ
        Kv = [
            [0., -1., 0.],
            [0., 1., 0.,],
            [0., 0., 0.]
        ]
        
        Kh = [
            [0., 0., 0.],
            [-1., 1., 0.],
            [0., 0., 0.]
        ]

        # フィルタ処理
        for y in range(H):
            for x in range(W):
                out_v[y + pad, x + pad] = np.sum(Kv * (tmp[y:y+self.K_size, x:x+self.K_size]))
                out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y:y+self.K_size, x:x+self.K_size]))
        
        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
        out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)

        self.differential_vimg = out_v
        self.differential_himg = out_h


img_file_path = "../imori.jpg"

process = process_class(img_file_path, K_size=3)

process.BGR2GRAY()
process.different_filter()

cv2.imwrite("./result/Question_14_v.jpg", process.differential_vimg)
cv2.imwrite("./result/Question_14_h.jpg", process.differential_himg)
#cv2.imshow("result v", process.differential_vimg)
# while cv2.waitKey(100) != 27: #loop if not get ESC
#     if cv2.getWindowProperty("result v", cv2.WND_PROP_VISIBLE) <= 0:
#         break
# cv2.destroyWindow("result v")

# cv2.imshow("result h", process.differential_himg)
# while cv2.waitKey(100) != 27: # loop if not get ESC or click x
#     if cv2.getWindowProperty("result h", cv2.WND_PROP_VISIBLE) <= 0:
#         break
# cv2.destroyWindow("result h")
# cv2.destroyAllWindows()


