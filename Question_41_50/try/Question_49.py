# オープニング処理を行う
# オープニング処理とはモルフォロジー処理
import cv2
import numpy as np

class molphorogy_class:
    def __init__(self, img_file_path):
        self.img = cv2.imread(img_file_path)
    
    def BGR2GRAY(self, img : np.array) -> np.array:
        b = img[:,:, 0].copy()
        g = img[:,:, 1].copy()
        r = img[:,:, 2].copy()

        gray = 0.2126*b + 0.1752*g + 0.0722*b

        gray = gray.astype(np.uint8)
        return gray

    def otsu_binarization(self, img:np.array, th=128) -> np.array:
        H, W = img.shape
        out = img.copy()

        max_sigma = 0
        max_t = 0

        # 閾値を定める処理
        for _t in range(1, 255):
            # 閾値よりも小さい画素値に対するパラメータの計算
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H*W)
            
            # 閾値よりも大きい画素値に対するパラメータの計算
            v1 = out[np.where(out >= _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H*W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)

            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t
        
        print("threshold >>", max_t)
        th = max_t
        out[out < th] = 0
        out[out >= th] = 255

        return out

    # モルフォロジー変換における縮小
    def Morphology_Erode(self, img:np.array, Erode_time=1) -> np.array:
        H, W = img.shape
        out = img.copy()

        MF = np.array((
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0)), dtype = np.uint8
        )

        # モルフォロジー処理を行う
        for i in range(Erode_time):
            tmp = np.pad(out, (1, 1), 'edge')
            for y in range(1, H+1):
                for x in range(1, W+1):
                    if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                        out[y-1, x-1] = 0
        
        return out
    
    # モルフォロジー変換における拡大
    # dilate -> 拡大する
    def Morphology_Dilate(self, img:np.array, Dilate_time=1) -> np.array:
        H, W = img.shape

        MF = np.array((
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0)), dtype=np.uint8
        )

        out = img.copy()
        for i in range(Dilate_time):
            tmp = np.pad(out, (1, 1), 'edge')
            for y in range(1, H+1):
                for x in range(1, W+1):
                    if np.sum(MF*tmp[y-1:y+2, x-1:x+2]) >= 255:
                        out[y-1, x-1] = 255

        return out
    
    def Morphology_Opening(self, img:np.array, time=1) -> np.array:
        out = self.Morphology_Erode(img, Erode_time=time)
        out = self.Morphology_Dilate(out, Dilate_time=time)
        return out
    
    def all_run(self) -> np.array:
        gray = self.BGR2GRAY(self.img)
        otsu = self.otsu_binarization(gray)
        out = self.Morphology_Opening(otsu, time=1)

        return out

img_file_path = "../imori.jpg"
process = molphorogy_class(img_file_path)
result = process.all_run()
cv2.imwrite("./result/Question_49.jpg", result)