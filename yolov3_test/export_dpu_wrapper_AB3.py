# import sys
# import numpy
# sys.modules['numpy._core'] = numpy.core
# import torch
# import torch.nn as nn
# from models.common import DetectMultiBackend
# from utils.torch_utils import select_device

# class YOLOv3DPUWrapper(nn.Module):
#     def __init__(self, weights_path, device='cpu'):
#         super().__init__()
#         # Convert string to torch.device object expected by DetectMultiBackend
#         device_obj = select_device(device)
#         base_model = DetectMultiBackend(weights_path, device=device_obj, fp16=False)
#         self.model = base_model.model
#         self.model.eval()

#         # Intercept the final Detect layer to return raw convolution outputs
#         detect_layer = self.model.model[-1]
        
#         def dpu_detect_forward(x):
#             # Pass each feature map through its respective Conv2d output head
#             return tuple(detect_layer.m[i](x[i]) for i in range(detect_layer.nl))

#         detect_layer.forward = dpu_detect_forward

#     def forward(self, x):
#         return self.model(x)

# if __name__ == "__main__":
#     wrapper = YOLOv3DPUWrapper("runs/train/exp2/weights/best.pt", device='cpu')
#     dummy_input = torch.randn(1, 3, 416, 416)
#     outputs = wrapper(dummy_input)
#     print("DPU-Safe Feature Map Shapes:")
#     for o in outputs:
#         print(o.shape)
#=============================================================================================================
# import sys
# import numpy

# # Only patch numpy._core if running on older NumPy 1.x (e.g., inside Vitis AI Docker)
# if not hasattr(numpy, '_core'):
#     try:
#         sys.modules['numpy._core'] = numpy.core
#     except AttributeError:
#         pass

# import torch
# import torch.nn as nn

# class YOLOv3DPUWrapper(nn.Module):
#     def __init__(self, weights_path, device='cpu'):
#         super().__init__()
#         # Allow custom class unpickling across PyTorch versions
#         try:
#             ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
#         except TypeError:
#             ckpt = torch.load(weights_path, map_location='cpu')
            
#         if isinstance(ckpt, dict):
#             model = ckpt.get('ema') or ckpt.get('model')
#         else:
#             model = ckpt
            
#         model = model.float().eval()
        
#         # Extract backbone and neck layers (excluding Detect head)
#         self.layers = nn.ModuleList([m for m in model.model[:-1]])
#         self.save = model.save
        
#         # Extract Detect layer indices and output heads
#         detect_layer = model.model[-1]
#         self.detect_f = detect_layer.f
#         self.m0 = detect_layer.m[0]
#         self.m1 = detect_layer.m[1]
#         self.m2 = detect_layer.m[2]

#     def forward(self, x):
#         y = []
#         for m in self.layers:
#             if m.f != -1:
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
#             x = m(x)
#             y.append(x if m.i in self.save else None)
        
#         # Route feature maps to prediction heads
#         head_inputs = [y[j] for j in self.detect_f]

#         out0 = self.m0(head_inputs[0])
#         out1 = self.m1(head_inputs[1])
#         out2 = self.m2(head_inputs[2])
        
#         return out0, out1, out2

# if __name__ == "__main__":
#     wrapper = YOLOv3DPUWrapper("runs/train/exp2/weights/best.pt", device='cpu')
#     dummy_input = torch.randn(1, 3, 416, 416)
#     outputs = wrapper(dummy_input)
#     print("DPU-Safe Feature Map Shapes:")
#     for o in outputs:
#         print(o.shape)
#==================================================================================================================
import os
# Prevent OpenMP and MKL thread allocation segmentation faults in Docker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Aliases numpy._core.* -> numpy.core.* so NumPy 2.x-pickled checkpoints load on
# the image's NumPy 1.x. Mapping only the parent segfaults torch.load; see
# vitis_compat/np2_pickle_compat.py. Requires vitis_compat on PYTHONPATH.
from np2_pickle_compat import apply as _apply_np2_compat

_apply_np2_compat()

import torch
import torch.nn as nn

torch.set_num_threads(1)

class YOLOv3DPUWrapper(nn.Module):
    def __init__(self, weights_path, device='cpu'):
        super().__init__()
        # Load weights safely without triggering C++ class reconstructors
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location='cpu')
            
        if isinstance(ckpt, dict):
            model = ckpt.get('ema') or ckpt.get('model')
        else:
            model = ckpt
            
        model = model.float().eval()
        
        self.layers = nn.ModuleList([m for m in model.model[:-1]])
        self.save = set(model.save)
        
        # Pre-calculate route mappings
        self.routes = []
        for m in self.layers:
            if isinstance(m.f, int):
                self.routes.append([m.f])
            else:
                self.routes.append(list(m.f))
                
        detect_layer = model.model[-1]
        self.detect_f = list(detect_layer.f)
        self.m0 = detect_layer.m[0]
        self.m1 = detect_layer.m[1]
        self.m2 = detect_layer.m[2]

    def forward(self, x):
        y = []
        for m, route in zip(self.layers, self.routes):
            if route == [-1]:
                x = m(x)
            else:
                layer_inputs = [x if j == -1 else y[j] for j in route]
                x = m(layer_inputs[0]) if len(layer_inputs) == 1 else m(layer_inputs)
            y.append(x if m.i in self.save else None)
        
        head_inputs = [y[j] for j in self.detect_f]
        out0 = self.m0(head_inputs[0])
        out1 = self.m1(head_inputs[1])
        out2 = self.m2(head_inputs[2])
        
        return out0, out1, out2

if __name__ == "__main__":
    wrapper = YOLOv3DPUWrapper("runs/train/exp2/weights/best.pt", device='cpu')
    dummy_input = torch.randn(1, 3, 416, 416)
    outputs = wrapper(dummy_input)
    print("DPU-Safe Feature Map Shapes:")
    for o in outputs:
        print(o.shape)