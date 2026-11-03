"""Đọc/ghi ảnh an toàn với đường dẫn có ký tự Unicode (ví dụ thư mục “ĐATN” trên Windows).

cv2.imread / cv2.imwrite trên Windows không đọc được đường dẫn có dấu tiếng Việt, nên mọi thao tác tệp
ảnh trong phần mềm đi qua numpy (fromfile / tofile) rồi mới giải mã / mã hóa bằng OpenCV.
"""
import os
import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, img, params=None):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf.tofile(path)
    return True
