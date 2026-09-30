"""Run a deterministic synthetic demonstration or compare a supplied image pair."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur_detection import FourierNetwork, block_scores, model_path, operation_counts


def synthetic_images(seed):
    rng = np.random.default_rng(seed)
    y, x = np.indices((512, 768))
    pixels = np.clip(128 + 35 * np.sin(x * 0.7) + 35 * np.cos(y * 0.6)
                     + rng.normal(0, 25, x.shape), 0, 255).astype("uint8")
    clean = Image.fromarray(pixels)
    defocus = clean.filter(ImageFilter.GaussianBlur(3))
    motion = Image.fromarray(np.mean([np.roll(pixels.astype(float), i, axis=1)
                                      for i in range(-10, 11)], axis=0).astype("uint8"))
    return {"clean": clean, "defocus": defocus, "motion": motion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=[128, 256], default=128)
    parser.add_argument("--k", type=int, default=72)
    parser.add_argument("--mask-radius", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", choices=["matched", "fft"], default="matched",
                        help="matched: each method uses its clean score; fft: paper-style shared FFT threshold")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--reference", type=Path, help="Clean calibration image")
    parser.add_argument("--image", type=Path, help="Corresponding image to evaluate")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.image):
        parser.error("Supply both --reference and --image")
    if args.k <= 0 or args.k > args.n or not (0 < args.mask_radius < args.n // 2):
        parser.error("Require 0 < K <= N and 0 < mask-radius < N/2")
    path = model_path(args.n, args.k)
    if not path.exists():
        parser.error(f"No pretrained model at {path}; choose an included width")
    network = FourierNetwork(path)
    if args.reference:
        with Image.open(args.reference) as source:
            reference = source.convert("L")
        with Image.open(args.image) as source:
            query = source.convert("L")
        images = {"clean": reference, "query": query}
        if reference.size != query.size:
            parser.error("Reference and query must have matching dimensions")
        source_name = "user-supplied pair"
    else:
        images = synthetic_images(args.seed)
        source_name = "synthetic illustration; not the paper evaluation dataset"
    args.output.mkdir(parents=True, exist_ok=True)
    threshold = float(block_scores(np.array(images["clean"]), args.n,
                                   mask_radius=args.mask_radius).mean())
    ann_clean_score = float(block_scores(np.array(images["clean"]), args.n,
                                        network, args.mask_radius).mean())
    ann_threshold = ann_clean_score if args.calibration == "matched" else threshold
    results = []
    canvas = Image.new("RGB", (320 * len(images), 300), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    for i, (name, image) in enumerate(images.items()):
        gray = np.asarray(image)
        exact = float(block_scores(gray, args.n, mask_radius=args.mask_radius).mean())
        approx = float(block_scores(gray, args.n, network, args.mask_radius).mean())
        row = {"image": name, "fft_score": exact, "ann_score": approx,
               "fft_blurry": exact < threshold, "ann_blurry": approx < ann_threshold,
               "ann_blurry_with_fft_threshold": approx < threshold}
        results.append(row)
        image.save(args.output / f"{name}.png")
        preview = image.convert("RGB"); preview.thumbnail((300, 210))
        canvas.paste(preview, (i * 320 + 10, 10))
        draw.text((i * 320 + 10, 225), name.title(), fill="#192b45")
        draw.text((i * 320 + 10, 245), f"FFT: {exact:.2f} | ANN: {approx:.2f}", fill="#192b45")
        draw.text((i * 320 + 10, 265),
                  "ANN decision: " + ("blurred" if row["ann_blurry"] else "not blurred"),
                  fill="#192b45")
    counts = operation_counts(args.n, args.k)
    baseline = sum(operation_counts(args.n, args.n).values())
    report = {"source": source_name, "seed": args.seed, "n": args.n, "k": args.k,
              "mask_radius": args.mask_radius, "calibration": args.calibration,
              "clean_fft_threshold": threshold, "ann_threshold": ann_threshold,
              "ann_operations_per_1d_call": counts,
              "reduction_against_untrimmed_ann_percent": 100 * (1 - sum(counts.values()) / baseline),
              "results": results}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    canvas.save(args.output / "comparison.png")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
