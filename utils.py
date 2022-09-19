import numpy as np
import os
import cv2


def save_data(data, path):
    data = data[0].cpu().permute(1, 2, 0).numpy()
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    data = np.uint8(data * 255)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, data)
