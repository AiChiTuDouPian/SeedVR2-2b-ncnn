import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import ncnn_io as io
with io.open_sd() as f:
    Wpin = io.get_top_weight(f, "vid_in.proj", "weight").astype(np.float32)
print("numpy Wpin range", Wpin.min(), Wpin.max(), "shape", Wpin.shape)
data = open('models/m5_graph/dit_block_bf16_0.bin','rb').read()
print("bin size", len(data))
# AdaCompose branch = 30720 float (假设 fp32) 或 fp16
nw = 132*2560; need = nw + 2560
for name, es in [("f32",4),("f16",2)]:
    acount = 1*2*6*2560  # AdaCompose 权重数
    for offset_elems in [acount, acount+0]:
        off_bytes = offset_elems * es
        if off_bytes + need*es <= len(data):
            arr = np.frombuffer(data[off_bytes:off_bytes+need*es], dtype=np.float32 if es==4 else np.float16)
            w = arr[:nw].reshape(2560,132); b = arr[nw:nw+2560]
            cos = float(np.sum(w*Wpin)/(np.linalg.norm(w)*np.linalg.norm(Wpin)+1e-12))
            print(f"[{name}] off_elems={offset_elems} cos={cos:.6f} w_range[{w.min():.4f},{w.max():.4f}] b_range[{b.min():.4f},{b.max():.4f}]")
