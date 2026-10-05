# HSV変換を実装して、色相Hを反転させる．そのあとRGBに戻す
import cv2
import numpy as np

class img_processor:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        self.hsv_img = None
        self.trans_rgb_img = None
    
    def BGR2HSV(self):
        img = self.img.copy().astype(np.float32) / 255.

        hsv = np.zeros_like(img, dtype=np.float32)

        # 全ての画素に対して最小値と最小値が含まれているチャンネル数を取り出している
        max_v = np.max(img, axis=2).copy()
        #print("max_v:", max_v)
        # max_v shape : (128, 128)
        min_v = np.min(img, axis=2).copy()
        min_arg = np.argmin(img, axis=2)
        print("min_arg shape: ", min_arg.shape)

        # HSV変換の定義
        # Hの変換式
        hsv[..., 0][np.where(max_v == min_v)]= 0
        
        ## if min == B
        ind = np.where(min_arg == 0)
        print("ind : ",ind)

        hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
        
        # if min == R
        ind = np.where(min_arg == 2)
        hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180

        # if min == G
        ind = np.where(min_arg == 1)
        hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

        # Sの変換式
        hsv[..., 1] = max_v.copy() - min_v.copy()

        # V
        hsv[..., 2] = max_v.copy()

        self.hsv_img = hsv
    
    def HSV2BGR(self):
        img = self.img.copy().astype(np.float32) / 255.

        # なんの最小と最大を取得しているの?
        max_v = np.max(img, axis=2).copy()
        min_v = np.min(img, axis=2).copy()

        out = np.zeros_like(img)

        if self.hsv_img is None:
            raise ValueError("HSV image is not computed. Please run BGR2HSV() first.")

        H = self.hsv_img[..., 0]
        S = self.hsv_img[..., 1]
        V = self.hsv_img[..., 2]

        C = S
        H_ = H / 60.
        X = C * (1 - np.abs(H_ % 2 - 1))
        Z = np.zeros_like(H)

        vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

        for i in range(6):
            ind = np.where((i <= H_) & (H_ < (i+1)))
            out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
            out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
            out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

        out[np.where(max_v == min_v)] = 0
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)

        self.trans_rgb_img = out


# 画像ファイルのパスを定義
img_file_path = "../imori.jpg"
processor = img_processor(img_file_path)
processor.BGR2HSV()

# hsv形式の画像を取得
hsv_img = processor.hsv_img

# HSV形式の画像を変換する
hsv_img[..., 0] = (hsv_img[..., 0] + 180) % 360
processor.hsv_img = hsv_img

processor.HSV2BGR()
trans_rgb_img = processor.trans_rgb_img
print("hsv_img shape: ", hsv_img.shape)
print("trans_rgb_img shape: ", trans_rgb_img.shape)

cv2.imwrite("./result/Question_05.jpg", trans_rgb_img)
#cv2.imshow("trans_rgb_img", trans_rgb_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

