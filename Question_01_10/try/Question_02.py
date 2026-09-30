"""Question 2: convert a BGR image to grayscale without OpenCV."""

import numpy as np
import cv2


def BGR2GRAY(img: np.ndarray) -> np.ndarray:
    """Return the luminance image calculated from a BGR image.

    The input uses the BGR channel ordering used by the exercise image.  The
    result is an unsigned 8-bit, two-dimensional grayscale image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must have shape (height, width, 3) in BGR order")
    #.transposeの役割は日記の2026_09_23.mdに書いてある
    b, g, r = img.astype(np.float64, copy=False).transpose(2, 0, 1)
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    #わかりにくいが戻り値は、もともとgrayを参照している
    #np.clip, np.floor, 1e-10, .astype(np.uint8)の役割は日記の2026_09_23.mdに書いてある
    return np.clip(np.floor(gray + 1e-10), 0, 255).astype(np.uint8)

#astypeは、numpy配列のデータ型を変換するためのメソッド
#astype(np.float)はnumpy配列のデータ型をfloat値に変換する
#通常のimreadでは画像の画素値をuint8型で読み込む
file_path = "../imori.jpg"
img = cv2.imread(file_path).astype(np.float64)

# グレースケール画像への変換
gray_img = BGR2GRAY(img)

# Save the result
cv2.imwrite("./result/Question_02.jpg", gray_img)
#cv2.imshow("result", gray_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
