import sys
import numpy

# NumPy 2.x pickles reference numpy._core; remap to numpy.core on NumPy 1.x (Vitis AI Docker)
if not hasattr(numpy, '_core'):
    sys.modules['numpy._core'] = numpy.core
    for _sub in ('multiarray', 'umath', 'numeric', '_multiarray_umath'):
        if hasattr(numpy.core, _sub):
            sys.modules['numpy._core.' + _sub] = getattr(numpy.core, _sub)

import torch
import yaml

def inspect_checkpoint(pt_path):
    print("\n" + "="*60)
    print(f" INSPECTING CHECKPOINT: {pt_path}")
    print("="*60)
    
    # Explicitly set weights_only=False to allow unpickling custom YOLO model classes
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)

    if not isinstance(ckpt, dict):
        print(f"[-] File loaded direct PyTorch module of type: {type(ckpt)}")
        return

    print(f"[+] Top-Level Keys in Checkpoint: {list(ckpt.keys())}")
    
    # 1. Metadata Inspection
    epoch = ckpt.get('epoch', 'N/A')
    print(f"[+] Saved Epoch: {epoch}")
    
    # 2. Extract Model Object
    model = ckpt.get('model', None)
    if model is not None:
        print(f"[+] Model Object Type: {type(model)}")
        
        # Extract Classes
        nc = getattr(model, 'nc', None)
        names = getattr(model, 'names', None)
        print(f"[+] Number of Classes (nc): {nc}")
        print(f"[+] Class Labels: {names}")
        
        # Extract Architecture YAML config if stored
        if hasattr(model, 'yaml') and model.yaml:
            print("\n--- Embedded Architecture YAML Config ---")
            print(yaml.dump(model.yaml))
                
        # Inspect Detect Layer & Anchors
        print("\n--- Detection Head & Anchors ---")
        for name, module in model.named_modules():
            if module.__class__.__name__ in ['Detect', 'Segment']:
                print(f"  Layer Name: {name} ({module.__class__.__name__})")
                if hasattr(module, 'anchors'):
                    print(f"  Anchors shape: {module.anchors.shape}")
                    print(f"  Anchors:\n{module.anchors}")
                print(f"  Strides: {getattr(module, 'stride', 'N/A')}")
                print(f"  Outputs per anchor (no): {getattr(module, 'no', 'N/A')}")

    # 3. Training Options
    opt = ckpt.get('opt', None)
    if opt:
        print("\n--- Saved Training Arguments ---")
        if isinstance(opt, dict):
            print(f"  imgsz: {opt.get('imgsz')}, batch_size: {opt.get('batch_size')}, data: {opt.get('data')}")
        else:
            print(f"  imgsz: {getattr(opt, 'imgsz', 'N/A')}, batch_size: {getattr(opt, 'batch_size', 'N/A')}")

if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or ['runs/train/exp2/weights/best.pt']
    for path in paths:
        inspect_checkpoint(path)
#==================================================================================================================
# import torch

# ckpt_path = "yolov3_merged23_e75.pt"
# print(f"Loading checkpoint: {ckpt_path}...\n")

# # Load checkpoint on CPU
# ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)

# print("==================================================")
# print("1. CHECKPOINT TOP-LEVEL KEYS")
# print("==================================================")
# if isinstance(ckpt, dict):
#     print("Keys found:", list(ckpt.keys()))
# else:
#     print("Checkpoint is a direct torch.nn.Module object.")

# print("\n==================================================")
# print("2. TRAINING ARGUMENTS & IMAGE SIZE")
# print("==================================================")
# # Search for training args/options
# opt = None
# if isinstance(ckpt, dict):
#     opt = ckpt.get("opt") or ckpt.get("train_args") or ckpt.get("args")

# if opt:
#     print("Found training arguments!")
#     if hasattr(opt, "__dict__"):
#         for k, v in vars(opt).items():
#             if any(x in k.lower() for x in ["img", "size", "batch", "data", "cfg", "weights", "nc"]):
#                 print(f"  --> {k}: {v}")
#     elif isinstance(opt, dict):
#         for k, v in opt.items():
#             if any(x in k.lower() for x in ["img", "size", "batch", "data", "cfg", "weights", "nc"]):
#                 print(f"  --> {k}: {v}")
# else:
#     print("No explicit training 'opt' dictionary saved in checkpoint.")
