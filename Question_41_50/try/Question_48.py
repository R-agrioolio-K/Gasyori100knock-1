# モルフォロジー変換の縮小を行う
import cv2
import numpy as np

class process_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
        print("img shape: ", self.img.shape)
    
    def BGR2GRAY(self, img):
        b = img[:,:, 0]
        g = img[:,:, 1]
        r = img[:,:, 2]

        gray = 0.2126*r + 0.7152*g + 0.0722*b

        gray = gray.astype(np.uint8)
        return gray
    
    # 大津の二値化
    def otsu_binarization(self, img, th=128):
        H, W = img.shape
        out = img.copy()

        # 閾値を保存するためのパラメータ
        max_sigma = 0
        max_t = 0

        for _t in range(1, 255):
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H*W)
            v1 = out[np.where(out > _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H*W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)
            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t
        
        # Binarization
        print("threshold >> ", max_t)
        th = max_t
        out[out < th] = 0
        out[out >= th] = 255

        return out
    
    # Morphology変換
    def Morphology_Erode(self, img, Dil_time=1):
        # 画像サイズの取得
        H, W = img.shape

        # フィルタの設定
        MF = np.array((
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0)
            ), dtype=np.uint8)
        
        # パディングの計算
        pad = MF.shape[0] // 2

        # モルフォロジー変換の実施
        out = img.copy()
        for i in range(Dil_time):
            tmp = np.zeros((H+pad*2, W+pad*2), dtype=np.uint8)
            tmp[1:H+pad, 1:W+pad] = out.copy()
            #print("tmp shape: ", tmp.shape)    #for debug
            for y in range(H):
                for x in range(W):
                    #print("tmp[y-1:y+2, x-1:x+2] shape", tmp[y:y+3, x:x+3].shape)  # for debug
                    if np.sum(MF*tmp[y:y+3, x:x+3]) < 255*4:
                        out[y-1, x-1] = 0
        
        return out
    
    def all_run(self):
        # グレースケール変換
        gray = self.BGR2GRAY(self.img)
        # 大津の二値化
        otsu = self.otsu_binarization(gray)
        # モルフォロジー変換
        out = self.Morphology_Erode(otsu, Dil_time=2)

        return out
    
img_file_path = "../imori.jpg"
process_ins = process_class(img_file_path)
result = process_ins.all_run()
cv2.imwrite("./result/Question_48.jpg", result)