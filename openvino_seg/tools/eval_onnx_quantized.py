import argparse
from pathlib import Path
import sys

sys.path.insert(0, ".")

import numpy as np
import onnxruntime as ort

from onnx_segmentation_utils import (
    load_resized_label,
    parse_hw,
    preprocess_image,
    read_annotation_pairs,
)


def make_session(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        # Auto-fallback checks
        candidates = [
            Path("tools/onnx") / model_path.name,
            Path("onnx") / model_path.name,
            Path(__file__).resolve().parent / "onnx" / model_path.name,
        ]
        for c in candidates:
            if c.is_file():
                model_path = c
                break
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    if input_meta.type != "tensor(float)":
        raise TypeError(f"{model_path} input must be tensor(float), got {input_meta.type}")
    return session, input_meta.name


def update_confusion(confusion, prediction, label, num_classes, ignore_label):
    valid = (label != ignore_label) & (label >= 0) & (label < num_classes)
    valid &= (prediction >= 0) & (prediction < num_classes)
    indices = num_classes * label[valid] + prediction[valid]
    confusion += np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_metrics(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denominator = true_positive + false_positive + false_negative
    ious = np.divide(true_positive, denominator, out=np.full_like(true_positive, np.nan), where=denominator != 0)
    pixel_accuracy = true_positive.sum() / max(confusion.sum(), 1)
    return {"ious": ious, "miou": np.nanmean(ious), "pixel_accuracy": pixel_accuracy}


def predict(session, input_name, image):
    """Read either class indices [N,1,H,W] or logits [N,C,H,W]."""
    output = session.run(None, {input_name: image})[0]
    if output.ndim != 4 or output.shape[0] != 1:
        raise ValueError(f"Expected [1,1,H,W] class indices or [1,C,H,W] logits, got {output.shape}")
    if output.shape[1] == 1:
        return output[0, 0].astype(np.int64, copy=False)
    return np.argmax(output[0], axis=0).astype(np.int64, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 and INT8 BlueFace ONNX segmentation accuracy")
    parser.add_argument("--fp32-path", default="./onnx/best-smi.onnx")
    parser.add_argument("--int8-path", default="./onnx/best_int8.onnx")
    parser.add_argument("--dataset-root", default=r'D:\F\ABlueFaceProj\20240618\Seg\BlueFaceDataX2', help="BlueFaceDataX2 root directory")
    parser.add_argument("--annotations", default="val.txt", help="Validation CSV, relative to --dataset-root when not absolute")
    parser.add_argument("--input-size", default="512,512", help="Fixed ONNX input size as H,W")
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--max-samples", type=int, default=0, help="Use 0 for all validation images")
    parser.add_argument("--max-miou-drop", type=float, default=0.005, help="Fail when FP32-INT8 mIoU exceeds this value")
    parser.add_argument("--min-pixel-agreement", type=float, default=0.995, help="Fail when FP32/INT8 pixel agreement is below this value")
    args = parser.parse_args()

    input_size = parse_hw(args.input_size)
    pairs = read_annotation_pairs(args.dataset_root, args.annotations)
    if args.max_samples:
        pairs = pairs[:args.max_samples]
    fp32_session, fp32_input = make_session(args.fp32_path)
    int8_session, int8_input = make_session(args.int8_path)
    fp32_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    int8_confusion = np.zeros_like(fp32_confusion)
    agreement_correct = 0
    agreement_total = 0

    for index, (image_path, label_path) in enumerate(pairs, 1):
        image = preprocess_image(image_path, input_size)
        label = load_resized_label(label_path, input_size)
        fp32_prediction = predict(fp32_session, fp32_input, image)
        int8_prediction = predict(int8_session, int8_input, image)
        update_confusion(fp32_confusion, fp32_prediction, label, args.num_classes, args.ignore_label)
        update_confusion(int8_confusion, int8_prediction, label, args.num_classes, args.ignore_label)
        valid = label != args.ignore_label
        agreement_correct += np.count_nonzero(fp32_prediction[valid] == int8_prediction[valid])
        agreement_total += np.count_nonzero(valid)
        if index == 1 or index % 25 == 0 or index == len(pairs):
            print(f"  Evaluated {index}/{len(pairs)}: {image_path.name}")

    fp32 = compute_metrics(fp32_confusion)
    int8 = compute_metrics(int8_confusion)
    agreement = agreement_correct / max(agreement_total, 1)
    print("\nModel       mIoU       Pixel Acc.")
    print(f"FP32        {fp32['miou']:.6f}   {fp32['pixel_accuracy']:.6f}")
    print(f"INT8        {int8['miou']:.6f}   {int8['pixel_accuracy']:.6f}")
    print(f"mIoU drop:  {fp32['miou'] - int8['miou']:.6f}")
    print(f"Pixel agreement (FP32 vs INT8): {agreement:.6f}")
    print("\nPer-class IoU")
    for class_id, (fp32_iou, int8_iou) in enumerate(zip(fp32["ious"], int8["ious"])):
        print(f"  class {class_id}: FP32={fp32_iou:.6f}  INT8={int8_iou:.6f}  drop={fp32_iou - int8_iou:.6f}")

    miou_drop = fp32["miou"] - int8["miou"]
    failures = []
    if miou_drop > args.max_miou_drop:
        failures.append(f"mIoU drop {miou_drop:.6f} > {args.max_miou_drop:.6f}")
    if agreement < args.min_pixel_agreement:
        failures.append(f"pixel agreement {agreement:.6f} < {args.min_pixel_agreement:.6f}")
    if failures:
        raise SystemExit("\nAccuracy gate: FAILED (" + "; ".join(failures) + ")")
    print("\nAccuracy gate: PASSED")


if __name__ == "__main__":
    main()
