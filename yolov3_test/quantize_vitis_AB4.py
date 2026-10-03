# import sys
# import numpy
# sys.modules['numpy._core'] = numpy.core
# import argparse
# import os
# import sys
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import sys
# import numpy

# if not hasattr(numpy, '_core'):
#     try:
#         sys.modules['numpy._core'] = numpy.core
#     except AttributeError:
#         pass
# # Verify Vitis AI pytorch_nndct availability
# try:
#     from pytorch_nndct.apis import torch_quantizer
# except ImportError:
#     print("[ERROR] 'pytorch_nndct' is missing! Run this script inside the Vitis AI PyTorch Docker environment (vitis-ai-pytorch).")
#     sys.exit(1)

# from export_dpu_wrapper_AB3 import YOLOv3DPUWrapper


# def get_calibration_dataloader(data_dir, img_size=416, max_samples=100):
#     """Loads and transforms calibration images safely into PyTorch FloatTensors."""
#     transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),  # Scales PIL Image (0-255) to FloatTensor (0.0-1.0) shape (3, H, W)
#     ])

#     valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
#     image_paths = []

#     for root, _, files in os.walk(data_dir):
#         for file in files:
#             if file.lower().endswith(valid_exts):
#                 image_paths.append(os.path.join(root, file))
#                 if len(image_paths) >= max_samples:
#                     break
#         if len(image_paths) >= max_samples:
#             break

#     if not image_paths:
#         raise FileNotFoundError(f"[ERROR] No valid images (.jpg, .png, etc.) found in directory: '{data_dir}'")

#     print(f"[INFO] Successfully found {len(image_paths)} calibration images in '{data_dir}'.")

#     calib_tensors = []
#     for path in image_paths:
#         try:
#             img = Image.open(path).convert('RGB')
#             tensor_img = transform(img)
#             calib_tensors.append(tensor_img)
#         except Exception as e:
#             print(f"[WARNING] Skipping unreadable image '{path}': {e}")

#     if not calib_tensors:
#         raise ValueError("[ERROR] Failed to load any valid calibration images.")

#     return calib_tensors


# def run_quantization(quant_mode, weights_path, calib_dir, output_dir, img_size=416):
#     os.makedirs(output_dir, exist_ok=True)

#     # Automatically select GPU if available, else CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Quantization running on device: {device}")

#     # 1. Load DPU Wrapper Model onto target device
#     print(f"[INFO] Initializing model wrapper with weights from '{weights_path}'...")
#     model = YOLOv3DPUWrapper(weights_path, device=device).to(device)
#     model.eval()

#     # 2. Prepare Dummy Input Tensor
#     dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32).to(device)

#     # 3. Instantiate Vitis AI Quantizer
#     # NOTE: input_args MUST be explicitly formatted as a 1-element tuple: (dummy_input,)
#     print(f"[INFO] Initializing Vitis AI Quantizer in '{quant_mode}' mode...")
#     quantizer = torch_quantizer(
#         quant_mode=quant_mode,
#         module=model,
#         input_args=(dummy_input,),
#         output_dir=output_dir
#     )

#     quant_model = quantizer.quant_model.to(device)
#     quant_model.eval()

#     # 4. Load Calibration Dataset
#     calib_images = get_calibration_dataloader(calib_dir, img_size=img_size)

#     # 5. Execute Forward Passes for Graph Tracing / Quantization
#     print(f"[INFO] Running forward pass for mode: '{quant_mode}'...")
#     with torch.no_grad():
#         for idx, img_tensor in enumerate(calib_images):
#             # Shape transformation: (3, H, W) -> (1, 3, H, W)
#             input_batch = img_tensor.unsqueeze(0).to(device)
#             _ = quant_model(input_batch)

#             # In 'test' mode, a single forward pass is enough to trace the graph for export
#             if quant_mode == 'test':
#                 print("[INFO] Single test forward pass complete.")
#                 break

#     # 6. Export Final Quantization Files / xmodel
#     if quant_mode == 'calib':
#         quantizer.export_quant_config()
#         print(f"\n[SUCCESS] Calibration finished! Scaling parameters saved to: '{output_dir}'")
#         print(f"[ACTION] Next, run with '--quant_mode test' to export the .xmodel file.")

#     elif quant_mode == 'test':
#         quantizer.export_xmodel(output_dir=output_dir, deploy=True)
#         print(f"\n[SUCCESS] Quantized .xmodel file exported successfully to: '{output_dir}'")
#         print(f"[ACTION] You can now compile the output .xmodel using 'vai_c_xir' for the VCK190.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Bulletproof Vitis AI Quantization for YOLOv3 VCK190")
#     parser.add_argument('--quant_mode', type=str, default='calib', choices=['calib', 'test'], help='Quantization mode: calib or test')
#     parser.add_argument('--weights', type=str, default='runs/train/exp2/weights/best.pt', help='Path to trained PyTorch weights (.pt)')
#     parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing calibration images')
#     parser.add_argument('--output_dir', type=str, default='quantize_result', help='Output directory for quantization files')
#     parser.add_argument('--img_size', type=int, default=416, help='Input image resolution (default: 416)')

#     args = parser.parse_args()

#     run_quantization(
#         quant_mode=args.quant_mode,
#         weights_path=args.weights,
#         calib_dir=args.data_dir,
#         output_dir=args.output_dir,
#         img_size=args.img_size
#     )
#=========================================================================================================================
# import sys
# import numpy

# # Safe NumPy patch for Vitis AI Docker (only runs if numpy._core is absent)
# if not hasattr(numpy, '_core'):
#     try:
#         sys.modules['numpy._core'] = numpy.core
#     except AttributeError:
#         pass

# import argparse
# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# try:
#     from pytorch_nndct.apis import torch_quantizer
# except ImportError:
#     print("[ERROR] 'pytorch_nndct' missing. Ensure you are running inside vitis-ai-pytorch.")
#     sys.exit(1)

# from export_dpu_wrapper_AB3 import YOLOv3DPUWrapper


# def get_calibration_dataloader(data_dir, img_size=416, max_samples=100):
#     transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#     ])

#     valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
#     image_paths = []

#     for root, _, files in os.walk(data_dir):
#         for file in files:
#             if file.lower().endswith(valid_exts):
#                 image_paths.append(os.path.join(root, file))
#                 if len(image_paths) >= max_samples:
#                     break
#         if len(image_paths) >= max_samples:
#             break

#     if not image_paths:
#         raise FileNotFoundError(f"[ERROR] No valid images found in: '{data_dir}'")

#     print(f"[INFO] Found {len(image_paths)} calibration images in '{data_dir}'.")

#     calib_tensors = []
#     for path in image_paths:
#         try:
#             img = Image.open(path).convert('RGB')
#             calib_tensors.append(transform(img))
#         except Exception as e:
#             print(f"[WARNING] Skipping image '{path}': {e}")

#     return calib_tensors


# def run_quantization(quant_mode, weights_path, calib_dir, output_dir, img_size=416):
#     os.makedirs(output_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Quantization running on device: {device}")

#     print(f"[INFO] Initializing DPU model wrapper from '{weights_path}'...")
#     model = YOLOv3DPUWrapper(weights_path, device=device).to(device)
#     model.eval()

#     dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32).to(device)

#     print(f"[INFO] Initializing Vitis AI Quantizer in '{quant_mode}' mode...")
#     quantizer = torch_quantizer(
#         quant_mode=quant_mode,
#         module=model,
#         input_args=(dummy_input,),
#         output_dir=output_dir
#     )

#     quant_model = quantizer.quant_model.to(device)
#     quant_model.eval()

#     calib_images = get_calibration_dataloader(calib_dir, img_size=img_size)

#     print(f"[INFO] Running forward passes for mode: '{quant_mode}'...")
#     with torch.no_grad():
#         for img_tensor in calib_images:
#             input_batch = img_tensor.unsqueeze(0).to(device)
#             _ = quant_model(input_batch)
#             if quant_mode == 'test':
#                 break

#     if quant_mode == 'calib':
#         quantizer.export_quant_config()
#         print(f"\n[SUCCESS] Calibration complete! Config saved to '{output_dir}'.")
#     elif quant_mode == 'test':
#         quantizer.export_xmodel(output_dir=output_dir, deploy=True)
#         print(f"\n[SUCCESS] Quantized .xmodel exported to '{output_dir}'.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--quant_mode', type=str, default='calib', choices=['calib', 'test'])
#     parser.add_argument('--weights', type=str, default='runs/train/exp2/weights/best.pt')
#     parser.add_argument('--data_dir', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, default='quantize_result')
#     parser.add_argument('--img_size', type=int, default=416)

#     args = parser.parse_args()
#     run_quantization(args.quant_mode, args.weights, args.data_dir, args.output_dir, args.img_size)
#==============================================================================================================
import os
# Prevent OpenMP / MKL multi-threading core dumps in Docker CPU mode
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

# Aliases numpy._core.* -> numpy.core.* so NumPy 2.x-pickled checkpoints load on
# the image's NumPy 1.x. Mapping only the parent segfaults torch.load; see
# vitis_compat/np2_pickle_compat.py. Requires vitis_compat on PYTHONPATH.
from np2_pickle_compat import apply as _apply_np2_compat

_apply_np2_compat()

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

torch.set_num_threads(1)

try:
    from pytorch_nndct.apis import torch_quantizer
except ImportError:
    print("[ERROR] 'pytorch_nndct' missing. Ensure you are running inside vitis-ai-pytorch.")
    sys.exit(1)

from export_dpu_wrapper_AB3 import YOLOv3DPUWrapper


def get_calibration_dataloader(data_dir, img_size=416, max_samples=100):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= max_samples:
                    break
        if len(image_paths) >= max_samples:
            break

    if not image_paths:
        raise FileNotFoundError(f"[ERROR] No valid images found in: '{data_dir}'")

    print(f"[INFO] Found {len(image_paths)} calibration images in '{data_dir}'.")

    calib_tensors = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            calib_tensors.append(transform(img))
        except Exception as e:
            print(f"[WARNING] Skipping image '{path}': {e}")

    return calib_tensors


def run_quantization(quant_mode, weights_path, calib_dir, output_dir, img_size=416):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    print(f"[INFO] Quantization running on device: {device}")

    print(f"[INFO] Initializing DPU model wrapper from '{weights_path}'...")
    model = YOLOv3DPUWrapper(weights_path, device=device).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32).to(device)

    print(f"[INFO] Initializing Vitis AI Quantizer in '{quant_mode}' mode...")
    quantizer = torch_quantizer(
        quant_mode=quant_mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=output_dir
    )

    quant_model = quantizer.quant_model.to(device)
    quant_model.eval()

    calib_images = get_calibration_dataloader(calib_dir, img_size=img_size)

    print(f"[INFO] Running forward passes for mode: '{quant_mode}'...")
    with torch.no_grad():
        for img_tensor in calib_images:
            input_batch = img_tensor.unsqueeze(0).to(device)
            _ = quant_model(input_batch)
            if quant_mode == 'test':
                break

    if quant_mode == 'calib':
        quantizer.export_quant_config()
        print(f"\n[SUCCESS] Calibration complete! Config saved to '{output_dir}'.")
    elif quant_mode == 'test':
        # This Vitis AI build's signature is export_xmodel(output_dir, deploy_check, dynamic_batch);
        # there is no `deploy` kwarg. deploy_check=True additionally dumps golden per-layer tensors
        # for on-board comparison - left off here to keep the disk footprint small.
        quantizer.export_xmodel(output_dir=output_dir, deploy_check=False)
        print(f"\n[SUCCESS] Quantized .xmodel exported to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_mode', type=str, default='calib', choices=['calib', 'test'])
    parser.add_argument('--weights', type=str, default='runs/train/exp2/weights/best.pt')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='quantize_result')
    parser.add_argument('--img_size', type=int, default=416)

    args = parser.parse_args()
    run_quantization(args.quant_mode, args.weights, args.data_dir, args.output_dir, args.img_size)