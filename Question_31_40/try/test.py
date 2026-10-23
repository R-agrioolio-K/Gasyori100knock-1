import cupy as cp

print("CUDA Available : ", cp.cuda.is_available())

if cp.cuda.is_available():

    device_count = cp.cuda.runtime.getDeviceCount()
    print("搭載GPU数 : ", device_count)
    for device_id in range(device_count):
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        print(f"---------- Device {device_id} ------")
        print(f"Device ID : {device_id}")
        print(f"Device Name : {props['name'].decode('utf-8')}")
    print("----------------------------")
    # 実際にGPU上で配列が作れるかテスト
    x = cp.array([1, 2, 3, 4, 5])
    print("Test Array:",x)
    print("Array Device : ", x.device)