# Synthetic AI Image Detection with Accuracy and Faithfulness Analysis

Research-grade pipeline for detecting AI-generated images and explaining the signals behind each prediction. This project combines a high-accuracy CNN detector with forensic feature models and produces interpretable visuals suitable for publications and presentations.

## Highlights
- **Dual detection**: CNN (ResNet-50) + forensic statistical features
- **Interpretability**: Grad-CAM for CNN and feature importance/SHAP for features
- **Robustness**: sensitivity to JPEG, resize, crop, and noise
- **Reproducible**: fixed seeds, saved models and plots

## Project Motivation
Generative models produce photorealistic images, which creates a need for reliable detection and transparent explanations. This repository focuses on both accuracy and faithfulness: detecting synthetic images and revealing the evidence used for classification.

## Methodology
1. **CNN detector** using pretrained ResNet-50 (binary classifier)
2. **Forensic feature detector** using frequency, noise, color, and texture statistics
3. **Faithfulness analysis** via Grad-CAM and feature importance/SHAP
4. **Robustness testing** under common image transformations

## Models Used
- ResNet-50 (PyTorch, transfer learning)
- Random Forest, SVM, Logistic Regression (scikit-learn)

## Feature Extraction Techniques
- FFT power spectrum, spectral entropy, high-frequency energy ratio
- Noise residual variance and spatial noise correlation
- RGB channel correlation
- Local Binary Patterns (LBP), edge density, gradient statistics

## Dataset Layout
Place images in the following structure:
```
data/
  real/
    img1.jpg
    ...
  synthetic/
    img1.jpg
    ...
```
Optional dataset download utilities are included for Stable Diffusion, StyleGAN, and ImageNet/COCO subsets. See `src/data_loader.py`.

## Visuals Gallery (Generated in `outputs/plots/`)
Run `python visualize.py` after placing data to generate a full set of visuals:
- `fft_example.png`: example FFT power spectrum
- `feature_high_freq_ratio.png`, `feature_spectral_entropy.png`, `feature_noise_variance.png`, `feature_noise_correlation.png`
- `feature_correlation_matrix.png`: correlation across forensic features
- `transformations_example.png`: JPEG/resize/crop/noise
- `gradcam_example.png`: CNN explanation (if CNN model exists)
- `feature_importance.png`, `shap_summary.png`: feature model explanations (if model exists)

## Results Comparison
The evaluation script outputs a comparison table:
```
Model | Accuracy | Precision | Recall | F1 | AUC
```
This includes CNN, forensic feature models, and a hybrid ensemble.

## Quick Start
Install dependencies:
```
pip install -r requirements.txt
```

Train the CNN detector:
```
python train_cnn.py
```

Train the forensic feature model:
```
python train_feature_model.py
```

Evaluate and compare models (including robustness tests):
```
python evaluate.py
```

Generate all visuals:
```
python visualize.py
```

Faithfulness / interpretability analysis only:
```
python faithfulness_analysis.py
```

For direct module execution without wrappers:
```
python -m src.train_cnn
python -m src.train_feature_model
python -m src.evaluate
python -m src.visualizations
python -m src.faithfulness_analysis
```

## Reproducibility
- Uses fixed random seeds
- Stores models in `outputs/models/`
- Stores plots and visualizations in `outputs/plots/`

## Project Structure
```
synthetic-ai-detection/
  data/
  notebooks/
  outputs/
  src/
  main.py
  requirements.txt
  README.md
```

## Notes
This code is structured for research presentations and GitHub readiness, with modular design, docstrings, and interpretable outputs.
