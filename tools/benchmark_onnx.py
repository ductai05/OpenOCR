#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Benchmark Script for SVTR vs SVTRv2 on Testset 30K (13k + 17k)
Optimized for ONNX Runtime (CPU / CUDA)
Supports independent evaluation of SVTR (v1) and SVTRv2 (v2).

Features:
- Independent model loading via CLI (--v1_path, --v2_path, or --model_path)
- Mode 1: High + Medium (27,772 samples standard benchmark)
- Mode 2: All qualities (High, Medium, Low, Bad hard-cases from CSV)
- Specific quality filtering (e.g. --quality low for difficult cases)
- Breakdown metrics by Dataset (13K, 17K) and by Quality (High, Medium, Low)
- Clean, clear metric reporting (Accuracy/EMR, CER, NED, Latency BS=1, Throughput FPS)
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime is not installed. Please run: pip install onnxruntime-gpu or pip install onnxruntime")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SVTR vs SVTRv2 ONNX Models on 30K LPR Testset")
    
    # Model arguments (support independent paths via CLI)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a single ONNX model to evaluate")
    parser.add_argument("--model_type", type=str, choices=["svtr", "svtrv2"], default=None,
                        help="Model architecture type: 'svtr' (BGR input) or 'svtrv2' (RGB input)")
    parser.add_argument("--v1_path", type=str, default=None,
                        help="Direct path to SVTR-Tiny (v1) ONNX model")
    parser.add_argument("--v2_path", type=str, default=None,
                        help="Direct path to SVTRv2-Tiny (v2) ONNX model")
    
    # Quality filter options (Option 1: high_medium, Option 2: all)
    parser.add_argument("--quality", "--quality_mode", dest="quality_mode", type=str, default="high_medium",
                        help="Quality filter: '1' or 'high_medium' (High+Medium, 27.7k), '2' or 'all' (Tất cả kể cả Low/Bad), 'low' (chỉ case khó Low), 'high', 'medium'")
    
    # Dataset & configuration
    parser.add_argument("--dataset_dir", type=str, default="D:/test_ocr_30k",
                        help="Root directory of 30K test set (containing ocr_test_13k and ocr_test_17k)")
    parser.add_argument("--eval_subset", type=str, choices=["all", "13k", "17k"], default="all",
                        help="Subset to evaluate: '13k', '17k', or 'all'")
    parser.add_argument("--dict_path", type=str, default=os.path.join(os.path.dirname(__file__), "dict_latin_38.txt"),
                        help="Path to Latin 38-character dictionary file")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for throughput and evaluation")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Inference device: 'cuda' or 'cpu'")
    parser.add_argument("--measure_latency", action="store_true", default=True,
                        help="Measure single-sample latency at batch_size=1")
    parser.add_argument("--latency_samples", type=int, default=300,
                        help="Number of iterations for BS=1 latency benchmark")
    parser.add_argument("--save_preds", type=str, default=None,
                        help="Optional file path to export predictions (e.g. preds.txt)")
    
    return parser.parse_args()


def load_character_dict(dict_path):
    """Load character dictionary with blank at index 0."""
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
    
    char_list = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\r\n")
            if not line:
                continue
            # Handle comma delimiter homoglyphs like 'H,Н' -> take primary 'H'
            char = line.split(",")[0]
            char_list.append(char)
            
    return char_list


def levenshtein_distance(s1: str, s2: str) -> int:
    """Fast dynamic-programming Levenshtein distance."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost_ins = prev[j + 1] + 1
            cost_del = curr[j] + 1
            cost_sub = prev[j] + (c1 != c2)
            curr.append(min(cost_ins, cost_del, cost_sub))
        prev = curr
    return prev[-1]


def normalize_quality_mode(mode_str):
    """Normalize user input quality mode."""
    m = str(mode_str).strip().lower()
    if m in ("1", "high_medium", "high+medium", "hm", "standard"):
        return "high_medium"
    elif m in ("2", "all", "all_quality", "tat_ca"):
        return "all"
    elif m in ("low", "hard", "kho"):
        return "low"
    elif m in ("high", "cao"):
        return "high"
    elif m in ("medium", "trung_binh"):
        return "medium"
    return m


def load_dataset_from_csv(csv_path, img_root, subset_name, allowed_qualities=None):
    """
    Load samples directly from CSV file with quality tags.
    """
    if not os.path.isfile(csv_path) or not os.path.isdir(img_root):
        print(f"[WARN] CSV or Image root missing for {subset_name}: {csv_path}")
        return []

    # Map quality dir names case-insensitively
    quality_dirs = {
        name.lower(): name
        for name in os.listdir(img_root)
        if os.path.isdir(os.path.join(img_root, name))
    }

    samples = []
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("Plate Text") or "").strip().upper()
            if not label:
                continue
            # Skip labels with uncertain character '_' (cannot score soundly)
            if "_" in label:
                continue

            quality_raw = (row.get("Plate Quality") or "Unknown").strip()
            quality_norm = quality_raw.lower()

            # Filter by quality if requested
            if allowed_qualities is not None and quality_norm not in allowed_qualities:
                continue

            filename = (row.get("Image") or "").strip()
            if not filename:
                continue

            subfolder = quality_dirs.get(quality_norm, quality_raw)
            # Find file on disk (.jpeg or .jpg)
            img_path = os.path.join(img_root, subfolder, f"{filename}.jpeg")
            if not os.path.isfile(img_path):
                img_path = os.path.join(img_root, subfolder, f"{filename}.jpg")
                if not os.path.isfile(img_path):
                    continue

            samples.append({
                "img_path": img_path,
                "label": label,
                "subset": subset_name,
                "quality": quality_raw.capitalize() if quality_raw else "Unknown"
            })

    return samples


def load_dataset(dataset_dir, eval_subset="all", quality_mode="high_medium"):
    """
    Load test samples according to requested subset and quality mode.
    """
    quality_mode = normalize_quality_mode(quality_mode)
    
    # Define allowed qualities
    if quality_mode == "high_medium":
        allowed_qualities = {"high", "medium"}
        mode_desc = "High + Medium (Chuẩn 27.7k mẫu)"
    elif quality_mode == "all":
        allowed_qualities = None  # No quality filtering: load all!
        mode_desc = "Tất cả các Quality (High, Medium, Low, Bad...)"
    elif quality_mode == "low":
        allowed_qualities = {"low"}
        mode_desc = "Chỉ các case khó (Low Quality)"
    elif quality_mode in ("high", "medium"):
        allowed_qualities = {quality_mode}
        mode_desc = f"Chỉ Quality: {quality_mode.capitalize()}"
    else:
        allowed_qualities = {quality_mode}
        mode_desc = f"Chỉ Quality: {quality_mode}"

    print(f"[INFO] Chế độ lọc Quality: {mode_desc}")

    # Paths configuration
    dir_13k = os.path.join(dataset_dir, "ocr_test_13k", "TestUS-CAN-MEX_Standard_13k_NewFormat_20260817")
    csv_13k = os.path.join(dir_13k, "PlateImage_Engine_data_13k_update.csv")
    label_txt_13k = os.path.join(dir_13k, "label_sets", "test_overall.txt")
    img_root_13k = os.path.join(dir_13k, "PlateImage_Engine_data_13k_1")

    dir_17k = os.path.join(dataset_dir, "ocr_test_17k", "Testset_Global")
    csv_17k = os.path.join(dir_17k, "PlateImage_Engine_data_17k_updated.csv")
    label_txt_17k = os.path.join(dir_17k, "label_sets", "test_overall.txt")
    img_root_17k = os.path.join(dir_17k, "TestUS-CAN-MEX_Standard_17k_NewFormat_20260817")

    all_samples = []

    # If high_medium and test_overall.txt exists, we can load directly for speed and exact 27,772 match
    if quality_mode == "high_medium":
        if eval_subset in ("all", "13k") and os.path.isfile(label_txt_13k):
            count_13k = 0
            with open(label_txt_13k, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        rel_path, label = parts[0], parts[1].strip().upper()
                        quality_tag = rel_path.split("/")[0] if "/" in rel_path else "High"
                        full_img_path = os.path.join(img_root_13k, rel_path.replace("/", os.sep))
                        all_samples.append({
                            "img_path": full_img_path,
                            "label": label,
                            "subset": "13k",
                            "quality": quality_tag
                        })
                        count_13k += 1
            print(f"[INFO] Loaded 13K subset (High+Med): {count_13k:,} samples")

        if eval_subset in ("all", "17k") and os.path.isfile(label_txt_17k):
            count_17k = 0
            with open(label_txt_17k, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        rel_path, label = parts[0], parts[1].strip().upper()
                        quality_tag = rel_path.split("/")[0] if "/" in rel_path else "High"
                        full_img_path = os.path.join(img_root_17k, rel_path.replace("/", os.sep))
                        all_samples.append({
                            "img_path": full_img_path,
                            "label": label,
                            "subset": "17k",
                            "quality": quality_tag
                        })
                        count_17k += 1
            print(f"[INFO] Loaded 17K subset (High+Med): {count_17k:,} samples")

    else:
        # Load from CSV for all qualities (including Low, Bad, etc.)
        if eval_subset in ("all", "13k"):
            samples_13k = load_dataset_from_csv(csv_13k, img_root_13k, "13k", allowed_qualities)
            all_samples.extend(samples_13k)
            print(f"[INFO] Loaded 13K subset ({quality_mode}): {len(samples_13k):,} samples")

        if eval_subset in ("all", "17k"):
            samples_17k = load_dataset_from_csv(csv_17k, img_root_17k, "17k", allowed_qualities)
            all_samples.extend(samples_17k)
            print(f"[INFO] Loaded 17K subset ({quality_mode}): {len(samples_17k):,} samples")

    print(f"[INFO] Total test samples to evaluate: {len(all_samples):,}\n")
    return all_samples


def create_onnx_session(onnx_path, device="cuda"):
    """Initialize ONNXRuntime session with GPU or CPU provider."""
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Optimize CPU threading: limit intra_op threads to prevent cache thrashing on Transformers
    cpu_cores = os.cpu_count() or 4
    sess_options.intra_op_num_threads = min(cpu_cores, 8)
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    providers = []
    if device == "cuda":
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        else:
            print("[WARN] CUDAExecutionProvider not available. Falling back to CPUExecutionProvider.")
    providers.append("CPUExecutionProvider")
    
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    
    # Query input and output node metadata
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    
    active_provider = session.get_providers()[0]
    return session, input_meta.name, output_meta.name, input_meta.shape, active_provider


def preprocess_image(img_path, model_type="svtr", in_channels=3, target_h=64, target_w=256):
    """
    Standard OCR preprocessing for SVTR / SVTRv2:
    - SVTR v1: BGR color space
    - SVTRv2 (3ch): RGB color space
    - SVTRv2 (1ch): Grayscale
    - Resize: [64, 256]
    - Norm: (x / 255.0 - 0.5) / 0.5
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    # Crucial Color Space Match
    if in_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif model_type == "svtrv2":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # else svtr v1 keeps OpenCV's default BGR
    
    # Direct resize without padding
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    if in_channels == 1:
        norm_img = resized.astype(np.float32)[np.newaxis, ...] / 255.0
    else:
        # Convert HWC to CHW float32 normalized in [-1, 1]
        norm_img = resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
    norm_img = (norm_img - 0.5) / 0.5
    return norm_img


def ctc_greedy_decode(preds_prob, char_list):
    """
    Greedy CTC Decoding:
    - Token 0 is blank (ignored)
    - Consecutive identical tokens collapsed
    - Remaining tokens mapped to char_list (1-indexed)
    """
    preds_idx = np.argmax(preds_prob, axis=-1)  # [B, T]
    batch_size = preds_idx.shape[0]
    decoded_texts = []
    
    for b in range(batch_size):
        seq = preds_idx[b]
        chars = []
        for t in range(len(seq)):
            idx = seq[t]
            if idx != 0 and (t == 0 or idx != seq[t - 1]):
                if 1 <= idx <= len(char_list):
                    chars.append(char_list[idx - 1])
        decoded_texts.append("".join(chars))
        
    return decoded_texts


def measure_single_latency(session, input_name, in_channels=3, target_h=64, target_w=256, num_samples=300):
    """Measure single-image inference latency (BS=1) in milliseconds."""
    dummy_input = np.random.randn(1, in_channels, target_h, target_w).astype(np.float32)
    
    # Warmup
    for _ in range(25):
        session.run(None, {input_name: dummy_input})
        
    latencies = []
    for _ in range(num_samples):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    return mean_lat, p50_lat, p95_lat


def evaluate_model(model_name, onnx_path, model_type, sample_list, char_list, args):
    """Run full benchmark for a single model across samples."""
    print("=" * 65)
    print(f"BENCHMARKING: {model_name}")
    print(f"  Model Path : {onnx_path}")
    print(f"  Model Type : {model_type} ({'RGB' if model_type == 'svtrv2' else 'BGR'} color mode)")
    
    session, input_name, output_name, input_shape, active_provider = create_onnx_session(onnx_path, args.device)
    print(f"  ONNX Runtime Provider: {active_provider}")
    print(f"  Input Node : {input_name} (Shape: {input_shape})")
    print(f"  Output Node: {output_name}")
    print("=" * 65)

    # Fair Batch Size Enforcement:
    batch_size = args.batch_size
    if isinstance(input_shape[0], int) and input_shape[0] > 0 and input_shape[0] != batch_size:
        print(f"\n[ATTENTION] Model '{model_name}' has FIXED input batch_size={input_shape[0]}, but --batch_size={batch_size} was requested.")
        print(f"  -> To benchmark fairly with --batch_size={batch_size}, please re-export this model with dynamic_axes (dynamo=False).")
        print(f"  -> Running strictly at batch_size={input_shape[0]} for this model as defined by its ONNX graph.\n")
        batch_size = input_shape[0]

    # Determine channel dimension (1 for grayscale v5, 3 for RGB v4/v1)
    model_channels = input_shape[1] if (len(input_shape) > 1 and isinstance(input_shape[1], int)) else 3
    print(f"  Detected Input Channels: {model_channels}")

    # 1. Latency benchmark (BS=1)
    latency_mean, latency_p50, latency_p95 = None, None, None
    if args.measure_latency:
        print("[1/2] Measuring Latency at Batch Size 1...")
        latency_mean, latency_p50, latency_p95 = measure_single_latency(
            session, input_name, in_channels=model_channels, target_h=64, target_w=256, num_samples=args.latency_samples
        )

    # 2. Accuracy & Throughput benchmark
    print(f"[2/2] Evaluating on {len(sample_list):,} samples (Batch Size = {batch_size})...")
    
    # Aggregators by subset (13k, 17k)
    subset_stats = {}
    # Aggregators by quality (High, Medium, Low, Bad, etc.)
    quality_stats = {}
    
    total_valid = 0
    total_correct = 0
    total_edit_dist = 0
    total_gt_chars = 0
    total_ned_sum = 0.0
    total_eval_time = 0.0
    exported_records = []

    pbar = tqdm(range(0, len(sample_list), batch_size), desc="Evaluating Batches", unit="batch")
    
    for idx in pbar:
        batch_slice = sample_list[idx : idx + batch_size]
        
        batch_imgs = []
        batch_meta = []
        for item in batch_slice:
            img_data = preprocess_image(item["img_path"], model_type=model_type, in_channels=model_channels)
            if img_data is not None:
                batch_imgs.append(img_data)
                batch_meta.append(item)
                
        if not batch_imgs:
            continue
            
        input_tensor = np.stack(batch_imgs, axis=0)
        
        # Inference timing
        t_start = time.perf_counter()
        outputs = session.run([output_name], {input_name: input_tensor})
        t_end = time.perf_counter()
        total_eval_time += (t_end - t_start)
        
        # CTC Decode
        preds = ctc_greedy_decode(outputs[0], char_list)
        
        # Score each prediction
        for pred_str, meta in zip(preds, batch_meta):
            gt_str = meta["label"]
            sub_name = meta["subset"]
            q_name = meta["quality"]

            # Initialize stats dicts
            if sub_name not in subset_stats:
                subset_stats[sub_name] = {"valid": 0, "correct": 0, "edit_dist": 0, "gt_chars": 0, "ned_sum": 0.0}
            if q_name not in quality_stats:
                quality_stats[q_name] = {"valid": 0, "correct": 0, "edit_dist": 0, "gt_chars": 0, "ned_sum": 0.0}

            is_correct = (pred_str == gt_str)
            dist = levenshtein_distance(pred_str, gt_str)
            gt_len = len(gt_str)
            max_len = max(len(pred_str), gt_len)
            ned = 1.0 - (dist / max_len) if max_len > 0 else 1.0

            # Update overall
            total_valid += 1
            if is_correct:
                total_correct += 1
            total_edit_dist += dist
            total_gt_chars += gt_len
            total_ned_sum += ned

            # Update subset
            subset_stats[sub_name]["valid"] += 1
            if is_correct:
                subset_stats[sub_name]["correct"] += 1
            subset_stats[sub_name]["edit_dist"] += dist
            subset_stats[sub_name]["gt_chars"] += gt_len
            subset_stats[sub_name]["ned_sum"] += ned

            # Update quality
            quality_stats[q_name]["valid"] += 1
            if is_correct:
                quality_stats[q_name]["correct"] += 1
            quality_stats[q_name]["edit_dist"] += dist
            quality_stats[q_name]["gt_chars"] += gt_len
            quality_stats[q_name]["ned_sum"] += ned

            if args.save_preds:
                exported_records.append(f"{gt_str}\t{pred_str}\t{'1' if is_correct else '0'}\t{sub_name}\t{q_name}\n")

    # Compute Overall
    overall_acc = (total_correct / total_valid * 100.0) if total_valid > 0 else 0.0
    overall_cer = (total_edit_dist / total_gt_chars * 100.0) if total_gt_chars > 0 else 0.0
    overall_ned = (total_ned_sum / total_valid * 100.0) if total_valid > 0 else 0.0
    throughput_fps = (total_valid / total_eval_time) if total_eval_time > 0 else 0.0

    # Save predictions if requested
    if args.save_preds:
        with open(args.save_preds, "w", encoding="utf-8") as f:
            f.writelines(exported_records)
        print(f"[INFO] Predictions written to: {args.save_preds}")

    # Output Clean Metrics (No tables, ready to take notes)
    print("\n" + "=" * 65)
    print(f"BENCHMARK RESULTS: {model_name}")
    print("=" * 65)
    
    # 1. Dataset Breakdown
    for sub_name, st in sorted(subset_stats.items()):
        acc = (st["correct"] / st["valid"] * 100.0) if st["valid"] > 0 else 0.0
        cer = (st["edit_dist"] / st["gt_chars"] * 100.0) if st["gt_chars"] > 0 else 0.0
        ned = (st["ned_sum"] / st["valid"] * 100.0) if st["valid"] > 0 else 0.0
        print(f"[{sub_name.upper()} SUBSET] ({st['valid']:,} samples)")
        print(f"  Accuracy (EMR) : {acc:.2f}%  ({st['correct']:,}/{st['valid']:,})")
        print(f"  CER            : {cer:.2f}%")
        print(f"  NED            : {ned:.2f}%\n")

    # 2. Quality Breakdown (Crucial for Low/Hard cases)
    if len(quality_stats) > 1:
        print("[BREAKDOWN BY QUALITY LEVEL]")
        for q_name, st in sorted(quality_stats.items()):
            acc = (st["correct"] / st["valid"] * 100.0) if st["valid"] > 0 else 0.0
            cer = (st["edit_dist"] / st["gt_chars"] * 100.0) if st["gt_chars"] > 0 else 0.0
            ned = (st["ned_sum"] / st["valid"] * 100.0) if st["valid"] > 0 else 0.0
            print(f"  * {q_name.upper():<7} ({st['valid']:>6,} samples) -> Acc: {acc:.2f}% | CER: {cer:.2f}% | NED: {ned:.2f}%")
        print()

    # 3. Overall
    print(f"[OVERALL EVALUATION] ({total_valid:,} samples)")
    print(f"  Accuracy (EMR) : {overall_acc:.2f}%  ({total_correct:,}/{total_valid:,})")
    print(f"  CER            : {overall_cer:.2f}%")
    print(f"  NED            : {overall_ned:.2f}%")
    if latency_mean is not None:
        print(f"  Latency (BS=1) : {latency_mean:.2f} ms  (P50: {latency_p50:.2f} ms, P95: {latency_p95:.2f} ms)")
    print(f"  Throughput FPS : {throughput_fps:,.1f} samples/s (BS={batch_size})")
    print("=" * 65 + "\n")


def main():
    args = parse_args()
    
    # Load character dictionary
    char_list = load_character_dict(args.dict_path)
    print(f"[INFO] Loaded dictionary with {len(char_list)} characters from {args.dict_path}")
    
    # Load test dataset with quality option
    samples = load_dataset(args.dataset_dir, eval_subset=args.eval_subset, quality_mode=args.quality_mode)
    if not samples:
        print("[ERROR] No valid samples found to evaluate. Exiting.")
        sys.exit(1)
        
    # Determine runs
    runs = []
    
    if args.v1_path:
        runs.append(("SVTR-Tiny (v1)", args.v1_path, "svtr"))
    if args.v2_path:
        runs.append(("SVTRv2-Tiny (v2)", args.v2_path, "svtrv2"))
        
    if args.model_path:
        model_type = args.model_type
        if model_type is None:
            model_type = "svtrv2" if "v2" in os.path.basename(args.model_path).lower() else "svtr"
            print(f"[INFO] Auto-detected model_type='{model_type}' from filename.")
        name = "SVTRv2-Tiny" if model_type == "svtrv2" else "SVTR-Tiny"
        runs.append((name, args.model_path, model_type))
        
    if not runs:
        print("[ERROR] No model path specified!")
        print("Usage examples:")
        print("  1. Chế độ 1 (High + Medium, 27.7k mẫu):")
        print("     python benchmark_onnx.py --quality 1 --v1_path \"C:\\Users\\GBF478\\Downloads\\svtr-training\\svtr_tiny_latin2m.onnx\"")
        print("  2. Chế độ 2 (Tất cả kể cả Low/Bad):")
        print("     python benchmark_onnx.py --quality 2 --v1_path \"C:\\Users\\GBF478\\Downloads\\svtr-training\\svtr_tiny_latin2m.onnx\"")
        print("  3. Đánh giá riêng case khó (Quality Low):")
        print("     python benchmark_onnx.py --quality low --v2_path \"C:\\Users\\GBF478\\Downloads\\svtrv2-training\\svtrv2_tiny.onnx\"")
        sys.exit(1)
        
    for model_name, path, m_type in runs:
        evaluate_model(model_name, path, m_type, samples, char_list, args)


if __name__ == "__main__":
    main()
