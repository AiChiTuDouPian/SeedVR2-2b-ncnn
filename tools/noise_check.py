import sys
import numpy as np
from PIL import Image

# 判定一张图是"自然图像"还是"纯噪声"：
#   lag-1 邻域相关系数：自然图 ~0.95+，白噪声 ~0
#   TV(总变差)/像素：自然图很小，白噪声很大
for p in sys.argv[1:]:
    try:
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0
    except Exception as e:
        print(f"{p:46s} ERROR {e}")
        continue
    x1 = a[:, :-1].ravel()
    x2 = a[:, 1:].ravel()
    ch = np.corrcoef(x1, x2)[0, 1]
    y1 = a[:-1, :].ravel()
    y2 = a[1:, :].ravel()
    cv = np.corrcoef(y1, y2)[0, 1]
    tv = (np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()) / 2
    verdict = "自然图像" if (ch > 0.85 and cv > 0.85) else ("噪声/损坏" if (ch < 0.5 or cv < 0.5) else "可疑")
    print(f"{p:46s} {a.shape[1]}x{a.shape[0]}  corrH={ch:+.4f} corrV={cv:+.4f} TV={tv:.4f} std={a.std():.4f}  -> {verdict}")
