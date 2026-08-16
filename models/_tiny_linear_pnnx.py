# pnnx model stat
# model inputshape = [1,3]f32
# FLOPS = 14
# memory OPS = 13

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Linear(bias=True, in_features=3, out_features=2)

        archive = zipfile.ZipFile('F:\Seedvr2\seedvr2-ncnn\models\_tiny_linear.pnnx.bin', 'r')
        self.fc.bias = self.load_pnnx_bin_as_parameter(archive, 'fc.bias', (2), 'float32')
        self.fc.weight = self.load_pnnx_bin_as_parameter(archive, 'fc.weight', (2,3), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.fc(v_0)
        return v_1

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("F:\Seedvr2\seedvr2-ncnn\models\_tiny_linear_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, dtype=torch.float)

    torch.onnx.export(net, v_0, "F:\Seedvr2\seedvr2-ncnn\models\_tiny_linear_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, dtype=torch.float)

    import pnnx
    pnnx.export(net, "F:\Seedvr2\seedvr2-ncnn\models\_tiny_linear_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
