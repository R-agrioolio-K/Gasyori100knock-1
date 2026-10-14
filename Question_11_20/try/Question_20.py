#画像のヒストグラムを実行するプログラム
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
img_file_path = "../imori_dark.jpg"
img = cv2.imread(img_file_path).astype(np.uint8)
H, W ,C= img.shape
print(f"H * W * C= {H*W*C}")

# ヒストグラムの処理
histgram = img.ravel()  #平坦化を行っている
print("histgram : ",histgram)
print(f"histgram_shape : {histgram.shape}")
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("./result/Question_20.png")