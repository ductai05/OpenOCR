#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe CPU-only ONNX Exporter for SVTRv2
Does NOT consume GPU VRAM - 100% safe to run while training is active.
"""

import os
import sys
import argparse
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from openrec.modeling import build_model
from openrec.postprocess import build_post_process


def parse_args():
    parser = argparse.ArgumentParser(description="Safe ONNX Export for SVTRv2")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to training config YAML")
    parser.add_argument("-p", "--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint (e.g. best.pth or best_snap.pth)")
    parser.add_argument("-o", "--output", type=str, default="./svtrv2_tiny.onnx",
                        help="Output .onnx path")
    parser.add_argument("--img_h", type=int, default=64, help="Input image height")
    parser.add_argument("--img_w", type=int, default=256, help="Input image width")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version (16+ required for TPS grid_sample)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    cfg = Config(args.config).cfg
    
    # 1. Setup vocabulary & character count
    post_process = build_post_process(cfg['PostProcess'], cfg['Global'])
    char_num = post_process.get_character_num()
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    
    # 2. Build model on CPU
    print("[1/3] Building SVTRv2 model on CPU...")
    model = build_model(cfg['Architecture'])
    model.eval()
    
    # 3. Load checkpoint safely on CPU (Zero GPU VRAM used)
    print(f"[2/3] Loading weights from: {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    # Strip DDP 'module.' prefix if present
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_dict[k[7:]] = v
        else:
            cleaned_dict[k] = v
            
    model.load_state_dict(cleaned_dict, strict=True)
    
    epoch_info = ckpt.get('epoch', 'N/A')
    step_info = ckpt.get('global_step', 'N/A')
    metrics_info = ckpt.get('metrics', {})
    print(f"      Loaded snapshot info: Epoch={epoch_info}, Step={step_info}")
    if metrics_info:
        acc = metrics_info.get('acc', 'N/A')
        ned = metrics_info.get('norm_edit_dis', 'N/A')
        print(f"      Recorded metric in ckpt: Acc={acc}, NED={ned}")
        
    # 4. Export to ONNX
    print(f"[3/3] Exporting to ONNX: {args.output}...")
    in_channels = cfg['Architecture'].get('in_channels', 3)
    dummy_input = torch.randn(1, in_channels, args.img_h, args.img_w, dtype=torch.float32, device='cpu')
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    
    try:
        # Use classic TorchScript exporter (dynamo=False) to pack all weights into a single standalone file
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 3: 'width'},
                'output': {0: 'batch_size'}
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 3: 'width'},
                'output': {0: 'batch_size'}
            },
            opset_version=args.opset,
            do_constant_folding=True
        )
    
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[SUCCESS] ONNX export complete: {args.output} ({file_size_mb:.2f} MB)")


if __name__ == '__main__':
    main()
