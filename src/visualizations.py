"""
Generate a broad set of visuals for the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.faithfulness_analysis import run_gradcam, feature_importance_plot, shap_summary_plot
from src.forensic_features import build_feature_matrix, collect_image_paths, _fft_power_spectrum
from src.preprocessing import apply_transformations


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_frequency_examples(paths: List[str], output_path: Path) -> None:
    img = Image.open(paths[0]).convert("L")
    power = _fft_power_spectrum(np.array(img))
    plt.figure(figsize=(4, 4))
    plt.imshow(np.log1p(power), cmap="magma")
    plt.title("FFT Power Spectrum (Example)")
    plt.axis("off")
    _save_figure(output_path)


def plot_feature_distributions(data_dir: str, output_dir: Path) -> None:
    paths, labels = collect_image_paths(data_dir)
    feature_result = build_feature_matrix(paths)
    df = pd.DataFrame(feature_result.features, columns=feature_result.names)
    df["label"] = labels

    for feature in ["high_freq_ratio", "spectral_entropy", "noise_variance", "noise_correlation"]:
        if feature in df.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df, x="label", y=feature)
            plt.title(f"{feature} by class")
            _save_figure(output_dir / f"feature_{feature}.png")

    # Correlation matrix
    plt.figure(figsize=(8, 6))
    corr = df.drop(columns=["label"]).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    _save_figure(output_dir / "feature_correlation_matrix.png")


def plot_transformations_example(data_dir: str, output_path: Path) -> None:
    paths, _ = collect_image_paths(data_dir)
    img = Image.open(paths[0]).convert("RGB")
    jpeg_img, resized_img, cropped_img, noisy_img = apply_transformations(img)

    plt.figure(figsize=(8, 6))
    for i, (title, im) in enumerate(
        [("original", img), ("jpeg", jpeg_img), ("resize", resized_img), ("crop", cropped_img), ("noise", noisy_img)]
    ):
        plt.subplot(2, 3, i + 1)
        plt.imshow(im)
        plt.title(title)
        plt.axis("off")
    _save_figure(output_path)


def generate_all_visuals(data_dir: str = "data", outputs_dir: str = "outputs") -> None:
    outputs_path = Path(outputs_dir) / "plots"
    outputs_path.mkdir(parents=True, exist_ok=True)

    paths, _ = collect_image_paths(data_dir)
    plot_frequency_examples(paths, outputs_path / "fft_example.png")
    plot_feature_distributions(data_dir, outputs_path)
    plot_transformations_example(data_dir, outputs_path / "transformations_example.png")

    cnn_path = Path(outputs_dir) / "models" / "cnn_resnet50.pt"
    forensic_path = Path(outputs_dir) / "models" / "random_forest_forensic.pkl"

    if cnn_path.exists():
        run_gradcam(paths[0], str(cnn_path), outputs_path / "gradcam_example.png")

    if forensic_path.exists():
        feature_importance_plot(str(forensic_path), data_dir, outputs_path / "feature_importance.png")
        shap_summary_plot(str(forensic_path), data_dir, outputs_path / "shap_summary.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project visualizations")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_visuals(data_dir=args.data_dir, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
