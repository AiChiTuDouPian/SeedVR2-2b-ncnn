import numpy as np, struct
f = open('latent_sr.bin','rb')
n = struct.unpack('<Q', f.read(8))[0]
H = struct.unpack('<i', f.read(4))[0]
W = struct.unpack('<i', f.read(4))[0]
d = np.frombuffer(f.read(), np.float32)
fin = np.isfinite(d)
print(f'H8={H} W8={W} n={n} floats={len(d)}')
print(f'range[{d[fin].min():.4f},{d[fin].max():.4f}] nan={np.isnan(d).sum()} inf={np.isinf(d).sum()}')
print(f'abs-mean={np.abs(d).mean():.4f}')
print(f'nonzero={np.count_nonzero(d)} / {d.size}')
