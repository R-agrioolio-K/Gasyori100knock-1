import torch
from torch import nn

loss = nn.CrossEntropyLoss()

N = 5 # バッチサイズ
K = 3 # クラス数
print(f"number of class: {K}")
y_pred = torch.randn(N, K, requires_grad=True)  #機械学習などで出力された予測値
print(f'予測値 {type(y_pred)} dtype={y_pred.dtype}')
print(f"予測値の中身: {y_pred}")
y_label = torch.empty(N, dtype=torch.long).random_(K)
print(f'正解値 {type(y_label)} {y_label.shape} dtype={y_label.dtype}')
print(f"正解値の中身: {y_label}")
output = loss(y_pred, y_label)
#output.backward()
print(f'output = {output.item()} dtype={output.dtype} shape={output.shape}')