import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import ncnn_io as io
# numpy proj 权重
with io.open_sd() as f:
    Wpin = io.get_top_weight(f, "vid_in.proj", "weight").astype(np.float32)
    Bpin = io.get_top_weight(f, "vid_in.proj", "bias").astype(np.float32)
print("numpy Wpin", Wpin.shape, "range", Wpin.min(), Wpin.max())
# block0 bin (ncnn 格式): 第一层 proj_vid_in 权重 132*2560 + bias 2560
# ncnn InnerProduct bin: weight (num_output, num_input) + bias
# 读取 bin 前 337920*4 + 2560*4 字节，尝试 fp32 或 fp16
data = open('models/m5_graph/dit_block_bf16_0.bin','rb').read()
print("bin size", len(data))
# 尝试 fp32
for dt, es in [('f4',4),('f2',2)]:
    n_w = 132*2560
    need = (n_w + 2560)*es
    if len(data) >= need:
        arr = np.frombuffer(data[:need], dtype=np.float32 if es==4 else np.float16)
        w = arr[:n_w].reshape(2560,132)
        b = arr[n_w:n_w+2560]
        print(f"[{dt}] proj w range", w.min(), w.max(), "bias range", b.min(), b.max())
        print(f"[{dt}] cos(w vs numpy)=", float(np.sum(w*Wpin)/(np.linalg.norm(w)*np.linalg.norm(Wpin)+1e-12)))
        print(f"[{dt}] w[0,0]=", w[0,0], " numpy Wpin[0,0]=", Wpin[0,0])
