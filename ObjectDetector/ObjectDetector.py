import cv2

def objectDetector():
    name_path = 'dataset\\coco.names'
    config_path = 'dataset\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weight_path = 'dataset\\frozen_inference_graph.pb'

    names = []
    with open(name_path, 'rt') as f:
        names = f.read().split('\n')

    # 直接调用库中的检测网络
    net = cv2.dnn_DetectionModel(weight_path, config_path)
    net.setInputSize(320, 520)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    return net, names