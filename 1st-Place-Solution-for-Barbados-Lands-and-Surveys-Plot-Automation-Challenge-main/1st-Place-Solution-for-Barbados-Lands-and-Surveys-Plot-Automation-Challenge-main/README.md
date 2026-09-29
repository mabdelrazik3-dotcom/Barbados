# 🏝️ Barbados Cadastral Plan Automation

> **Automated extraction of land parcel geometries and metadata from analog survey plans using Vision-Language Models and Deep Learning**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/Zindi-Competition-orange.svg)](https://zindi.africa/competitions/barbados-lands-and-surveys-plot-automation-challenge/leaderboard)

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏆 Challenge Context](#-challenge-context)
- [⚡ INFERENCE ONLY (Quick Start) ⭐](#-inference-only-quick-start-)
- [📁 Directory Structure](#-directory-structure)
- [🏗️ Solution Architecture](#️-solution-architecture)
  - [Segmentation Pipeline](#1-segmentation-pipeline-)
  - [Text Extraction Pipeline](#2-text-extraction-pipeline-)
- [💻 Hardware Requirements](#-hardware-requirements)
- [🛠️ Installation & Environment Setup](#️-installation--environment-setup)
- [🎓 Full Training Pipeline](#-full-training-pipeline)
  - [Segmentation Training](#segmentation-training-)
  - [Text Extraction Training](#text-extraction-training-)
- [🏆 Competition Results](#-competition-results)
- [🔗 Key Technologies](#-key-technologies)
- [📧 Contact](#-contact)

---

## 🎯 Project Overview

This solution automates the digitization of cadastral survey plans for the **Barbados Lands and Surveys Department**. It combines **state-of-the-art Vision-Language Models** with **semantic segmentation** to extract:

- 🗺️ **Land parcel polygons** (pixel coordinates)
- 📄 **Metadata fields**: Land Surveyor, Surveyed For, Address, Certified Date, Total Area, Unit of Measurement, Parish, LT Number

The system processes ~700 training plans and ~300 test plans, producing a final submission CSV with geometries and metadata ready for digital registry integration.

### ✨ Key Highlights

- 🤖 **VLM-Powered Alignment**: Qwen3-VL-30B (32B) reasoning model aligns geographic coordinates to pixel space
- 🧠 **Fine-Tuned OCR**: Qwen3-VL-8B trained with Unsloth LoRA for metadata extraction
- 🔄 **Auto-Correction**: VLM-based label correction + pseudo-labeling for robust training
- 🎯 **Bias Regularization**: Surveyor-specific embeddings improve edge case handling- Naming conventions, how middle name initials and geometries are preferred by various surveyors and LT Nums
- 🔁 **Test-Time Augmentation**: 8-way TTA for robust segmentation predictions

---

## 🏆 Challenge Context

**Challenge**: Barbados Cadastral Plan Digitization
**Platform**: Zindi Africa
**Goal**: Modernize land record handling by automating extraction from analog survey plans

### 📊 Evaluation Metrics

The solution is evaluated using a **weighted multi-metric system**:

| Metric | Weight | Description |
|--------|--------|-------------|
| **IoU Polygon** | 0.5 | Intersection over Union for land parcel shapes |
| **Word Error Rate (WER)** | 0.2 | Accuracy of extracted text (TargetSurvey field) |
| **Multi-Column Accuracy (MCA)** | 0.3 | Exact match across 5 metadata fields |

**Final Score** = `(0.5 × IoU) + (0.2 × (1 - WER/2)) + (0.3 × MCA)`

### 📝 Target Fields

**TargetSurvey** (for WER):

- Concatenation: `Land Surveyor + " " + Surveyed For + " " + Address`
- Lowercase, punctuation removed, whitespace normalized

**Multi-Column Accuracy** (MCA) fields:

- Certified date
- Total Area
- Unit of Measurement
- Parish
- LT Num

---

## ⚡ INFERENCE ONLY (Quick Start) ⭐

```text
╔═══════════════════════════════════════════════════════════════════════╗
║  🎯 Want to run inference on new data without training? Start here!  ║
║                                                                       ║
║  This section is for inference using pre-trained model checkpoints    ║
║  and if you want to generate predictions on new survey plans.               ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### **📋 Prerequisites**

- Pre-trained model checkpoints (included in the provided package):
  - `outputs/seg_model_checkpoints/` (segmentation model)
  - `outputs/bias_model_checkpoints/bias_model.pt` (bias model)
  - `outputs/qwen3vl8b_finetuned_lora/` (fine-tuned LoRA adapter)
  - `models/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`(full model to be used with the finetuned LoRA adapter)

### **Step 1: Prepare Environment** 🌱

```bash
# Extract the provided ZIP file
unzip Barbados2.zip
cd Barbados2

# Install dependencies
bash install.sh

# Activate environment
source .venv/bin/activate
```

### **Step 2: Prepare Data** 📦

Place your test data in the correct locations:

```bash
Barbados2/
├── data/
│   ├── Test.csv                    # Test sample IDs and metadata
│   └── survey_plans/               # Test images (.jpg)
│       ├── 7703-078.jpg
│       ├── 7704-081.jpg
│       └── ...
```

### **Step 3: Update Configuration** ⚙️

Edit `configs/base.yaml` with your paths:

```yaml
root_dir: /path/to/Barbados2
test_images_dir: /path/to/Barbados2/data/survey_plans
test_csv_path: /path/to/Barbados2/data/Test.csv

seed: 123
folds: 5
val_fold: 2
```

### **Step 4: Run Inference** 🎯

```bash
bash run_inference.sh
```

**What Happens**:

1. **Text Extraction**
   - Patchify test images (7 crops each)
   - Apply fine-tuned Qwen model
   - Clean and format predictions
   - Output: `data/sub_text_extraction.csv`

2. **Segmentation**
   - Load segmentation model + bias embeddings
   - Run 8-way TTA for each image
   - Extract polygons from masks
   - Merge with text metadata
   - **Output**: `data/final_submission.csv` ✅

**Total Runtime**: ~1 hour for 219 test images (1× RTX A6000)

### **Step 5: Retrieve Results** 📊

Your final submission is ready at:

```
data/final_submission.csv
```

**Format**:

```csv
ID,TargetSurvey,Certified date,Total Area,Unit of Measurement,Parish,LT Num,geometry
7703-078,andre clarke d & a developers ltd lot 1 foul bay,2013-11-22,411.0,sq m,St. Philip,77.03.08.014,"[(40621.893, 66595.872), ...]"
```

---

## 📁 Directory Structure

```
Barbados2/
│
├── 📂 configs/                          # Configuration files
│   ├── base.yaml                        # Root paths, seed, fold config
│   ├── segmentation.yaml                # Segmentation hyperparameters
│   └── text_extraction.yaml             # VLM model configurations
│
├── 📂 data/                             # Training/test data and outputs
│   ├── Train.csv                        # ~700 training samples with targets
│   ├── Test.csv                         # ~300 test samples (no labels)
│   ├── SampleSubmission.csv             # Submission format template
│   ├── survey_plans/                    # Input images (.jpg)
│   ├── df.csv                           # Merged dataset with K-fold splits
│   ├── geom_px_df.csv                   # Aligned pixel-space polygons
│   ├── label_corrections.csv            # VLM-corrected metadata
│   └── final_submission.csv             # 🎯 Final submission file
│
├── 📂 models/                           # Pre-trained model checkpoints
│   ├── Qwen3-VL-30B-A3B-Thinking/       # Reasoning VLM for alignment
│   └── Qwen3-VL-8B-Instruct-unsloth-bnb-4bit/  # Fine-tunable OCR model
│
├── 📂 outputs/                          # Trained model checkpoints
│   ├── seg_model_checkpoints/           # Segmentation model weights
│   ├── bias_model_checkpoints/          # Surveyor bias embeddings
│   └── qwen3vl8b_finetuned_lora/        # Fine-tuned LoRA adapter
│
├── 📂 preprocessing/
│   └── create_dataset.py                # Assemble dataset with K-fold splits
│
├── 📂 segmentation/
│   ├── 📂 align_polys/                  # VLM-powered geometry alignment
│   │   ├── create_qwen_env.sh           # Setup isolated vLLM environment
│   │   ├── run_qwen_vllm.sh             # Launch Qwen vLLM server
│   │   └── align_geo_coords_to_px.py    # Geo → pixel coordinate conversion
│   ├── 📂 train/
│   │   ├── train_seg.py                 # Lightning Unet++ training
│   │   └── train_bias.py                # Surveyor bias embedding training
│   └── 📂 infer/
│       └── seg_inference.py             # TTA segmentation + final submission
│
├── 📂 text_extraction/
│   ├── 📂 train/
│   │   ├── download_models.py           # Download Qwen checkpoints
│   │   ├── patchify_images.py           # Tile images (7 crops per image)
│   │   ├── automated_label_correction.py # VLM label correction
│   │   ├── train_and_create_pseudo.py   # First fine-tune + pseudo labels
│   │   └── train_with_pseudos.py        # Final fine-tune with all data
│   ├── 📂 infer/
│   │   ├── text_inference.py            # VLM metadata extraction
│   │   └── clean_text_preds.py          # Post-processing & formatting
│   └── prompts.py                       # VLM prompt templates
│
├── 📂 utils/
│   ├── base_utils.py                    # Config, WKT parsing, alignment
│   ├── seg_utils.py                     # Bias loading, polygon metrics
│   └── text_utils.py                    # Reproducibility helpers
│
├── 🔧 install.sh                        # Bootstrap main environment
├── 🚀 run_seg_training.sh               # Orchestrate segmentation training
├── 🚀 run_text_training.sh              # Orchestrate text extraction training
├── 🚀 run_inference.sh                  # Full inference pipeline
└── 📄 requirements.txt                  # Python dependencies
```

---

## 🏗️ Solution Architecture

The solution is divided into **two parallel pipelines** that converge at inference:

```
                ╔═══════════════════════════════════════════════════════════╗
                ║              📊 DATA PREPROCESSING                        ║
                ║   ┌───────────────────────────────────────────────────┐   ║
                ║   │  create_dataset.py → df.csv (K-fold splits)       │   ║
                ║   └───────────────────────────────────────────────────┘   ║
                ╚═══════════════════════════════════════════════════════════╝
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
       ╔═════════════════▼═════════════╗   ╔════════════▼═══════════════════╗
       ║  🗺️  SEGMENTATION PIPELINE   ║   ║  📄 TEXT EXTRACTION PIPELINE   ║
       ║                               ║   ║                                ║
       ║  1. Geo→Pixel Alignment       ║   ║  1. Download Models            ║
       ║     (Qwen 32B VLM)            ║   ║  2. Patchify Images            ║
       ║  2. Train Unet++              ║   ║  3. Label Correction           ║
       ║     (EfficientNet-B5)         ║   ║  4. Fine-tune + Pseudos        ║
       ║  3. Train Bias Model          ║   ║  5. Final Fine-tune            ║
       ║                               ║   ║                                ║
       ╚═══════════════╤═══════════════╝   ╚════════════╤═══════════════════╝
                       │                                │
                       │              ┌─────────────────▼─────────────────┐
                       │              │  📝 Text Inference                │
                       │              │  → sub_text_extraction.csv        │
                       │              └─────────────────┬─────────────────┘
                       │                                │
                       └────────────────┬───────────────┘
                                        │
                      ┌─────────────────▼─────────────────┐
                      │  🎯 Segmentation Inference        │
                      │  (TTA + Bias Blending)            │
                      │  → final_submission.csv ✅        │
                      └───────────────────────────────────┘
```

---

### **1. Segmentation Pipeline** 🗺️

Extracts land parcel polygon coordinates using semantic segmentation with surveyor bias regularization.

#### **🔧 Training Phase**

##### **Step 1: Geometry Alignment** 🤖

**Challenge**: Survey plans contain geographic coordinates that must be mapped to pixel space for training.

**Solution**: Use **Qwen3-VL-30B** (32B parameter reasoning VLM) to align coordinates.

- **Script**: `segmentation/align_polys/align_geo_coords_to_px.py`
- **Method**:
  1. Launch local vLLM server (`run_qwen_vllm.sh`) with 4-way tensor parallelism
  2. Send tiled survey plan images + coordinate hints to VLM
  3. Model reasons about coordinate positions and returns pixel mappings
- **Output**: `data/geom_px_df.csv` (aligned pixel-space polygons)
- ⚠️ **Note**: This process is **non-deterministic** due to the stochastic nature of VLMs

##### **Step 2: Segmentation Model Training** 🧠

**Architecture**: **Unet++** with **EfficientNet-B5** encoder (Segmentation Models PyTorch)

- **Script**: `segmentation/train/train_seg.py`
- **Framework**: PyTorch Lightning
- **Training Details**:
  - **Input**: 2048×2048 images (resized from originals - aspect ratio kept)
  - **Augmentation**: Heavy (rotation ±90°, flips, perspective, elastic deformation, optical distortion)
  - **Loss**: IoU (Jaccard) loss
  - **Optimizer**: Adam (lr=0.0003)
  - **Precision**: 16-bit mixed precision (AMP)
  - **Validation**: 5-fold cross-validation (fold 2 for validation)
  - **Epochs**: 120 max with early stopping
- **Output**: Lightning checkpoint → `outputs/seg_model_checkpoints/`

##### **Step 3: Bias Model Training** 🎯

**Purpose**: Capture surveyor-specific naming conventions and geometry patterns to regularize predictions.

- **Script**: `segmentation/train/train_bias.py`
- **Method**: Creates per-LT-number (surveyor ID) embeddings encoding:
  - Typical polygon shapes for that surveyor
  - Naming conventions and metadata patterns
- **Usage**: During inference, provides fallback geometries when model confidence is low
- **Output**: `outputs/bias_model_checkpoints/bias_model.pt`

#### **📡 Inference Phase**

**Script**: `segmentation/infer/seg_inference.py`

**Pipeline**:

1. Load trained Unet++ model and bias embeddings
2. **Test-Time Augmentation (TTA)**: 8-way augmentation (scale + flip variations)
3. Ensemble TTA predictions:
   - Compute IoU between all polygon pairs
   - Choose polygon with highest average IoU (consistency)
   - Verify smoothness (lowest perimeter-to-area ratio)
4. Apply bias-guided geometry refinement for low-confidence predictions
5. Convert binary masks → polygons via contour extraction + Douglas-Peucker smoothing
6. Merge with text extraction metadata
7. **Output**: `data/final_submission.csv` ✅

---

### **2. Text Extraction Pipeline** 📄

Extracts metadata fields using fine-tuned Vision-Language Models with auto-correction and pseudo-labeling.

#### **🔧 Training Phase**

##### **Step 1: Download Models** 📥

- **Script**: `text_extraction/train/download_models.py`
- **Models Downloaded**:
  - `Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` (4-bit quantized for fine-tuning)
  - Supporting checkpoints for Unsloth LoRA training

##### **Step 2: Image Patchification** ✂️

**Motivation**: Survey plans are large (3000×4000px); tiling improves VLM focus on relevant regions.

- **Script**: `text_extraction/train/patchify_images.py`
- **Method**: Creates **7 crops per image** (1024×1024):
  - Full image (resized)
  - Top half
  - Bottom half
  - 4 quadrants (top-left, top-right, bottom-left, bottom-right)
- **Output**:
  - `data/patched_1024.csv` (training)
  - `data/test_patched_1024.csv` (test)
  - `data/patched_images/` and `data/test_patched_images/`

##### **Step 3: Automated Label Correction** 🔍

**Challenge**: Training labels contain OCR errors and inconsistencies.

**Solution**: Use Qwen3-VL-8B (thinking mode) to auto-correct noisy labels.

- **Script**: `text_extraction/train/automated_label_correction.py`
- **Method**:
  1. Send image patches + raw labels to VLM
  2. Specialized prompt guides model to verify/correct each field
  3. Model outputs corrected JSON metadata
- **Output**: `data/label_corrections.csv` (cleaned training labels)

##### **Step 4: First Fine-Tune + Pseudo Label Generation** 🎓

- **Script**: `text_extraction/train/train_and_create_pseudo.py`
- **Method**:
  1. Fine-tune Qwen3-VL-8B on corrected training labels (Unsloth LoRA)
  2. Apply fine-tuned model to test data → generate pseudo labels
- **Output**:
  - Intermediate LoRA adapter
  - `data/pseudo_df.csv` (pseudo labels for test data)

##### **Step 5: Final Fine-Tune (Train + Pseudo)** 🚀

- **Script**: `text_extraction/train/train_with_pseudos.py`
- **Method**: Retrain on **union** of corrected train labels + pseudo test labels
- **Benefit**: Improves generalization and reduces distribution shift
- **Output**: `outputs/qwen3vl8b_finetuned_lora/` (final inference adapter)

#### **📡 Inference Phase**

##### **Step 1: Text Prediction** 📝

- **Script**: `text_extraction/infer/text_inference.py`
- **Method**:
  1. Patchify test images (same 7-crop layout)
  2. Apply fine-tuned Qwen model to each patch
  3. Extract JSON metadata fields
- **Output**: `data/text_predictions_df.csv` (raw predictions)

##### **Step 2: Post-Processing & Cleaning** 🧹

- **Script**: `text_extraction/infer/clean_text_preds.py`
- **Operations**:
  - Normalize parish names (e.g., "St Philip" → "St. Philip")
  - Standardize units ("square meters" → "sq m")
  - Clean addresses (lowercase, remove punctuation)
  - Construct **TargetSurvey** column: `Land Surveyor + Surveyed For + Address` (cleaned)
- **Output**: `data/sub_text_extraction.csv` (cleaned metadata)

---

## 💻 Hardware Requirements

### **🏋️ Training Environment**

| Component | Specification |
|-----------|---------------|
| **GPU** | 4× NVIDIA RTX A6000 (48GB VRAM each) |
| **RAM** | 256 GB |
| **CPU** | 32 cores |
| **Storage** | 2 TB (HDD/SSD) |
| **Runtime** | ~16 hours total |

**Runtime Breakdown**:

- Qwen alignment: ~14 hours (dominant bottleneck)
- Segmentation training: ~1.5 hours
- Text extraction training: ~0.5 hours

### **⚡ Inference Environment**

| Component | Specification |
|-----------|---------------|
| **GPU** | 1× NVIDIA RTX A6000 (48GB VRAM) |
| **RAM** | 64 GB |
| **CPU** | 16 cores |
| **Runtime** | ~1 hour for 219 test images |

**Runtime Breakdown**:

- Text extraction: ~40 minutes
- Segmentation: ~20 minutes

---

## 🛠️ Installation & Environment Setup

### **1. Install Main Environment** 🐍

```bash
# Extract the provided ZIP file
unzip Barbados2.zip
cd Barbados2

# Install main environment
bash install.sh

# Activate environment
source .venv/bin/activate
```

**System Requirements**:

- Ubuntu 22.04 LTS (recommended)
- Python 3.11+
- CUDA 12.4

**Creates**: `.venv/` with:

- PyTorch 2.5.1 (CUDA 12.4)
- PyTorch Lightning
- Unsloth (for LoRA fine-tuning)
- vLLM 0.10.0
- Segmentation Models PyTorch (SMP)
- Transformers 4.57.0
- GeoPandas, Shapely, OpenCV, Albumentations

### **2. Install Qwen Alignment Environment** 🤖

*(Only required for geometry alignment during training)*

```bash
bash segmentation/align_polys/create_qwen_env.sh
```

**Creates**: `.qvenv/` with:

- vLLM 0.11.0 (isolated version for Qwen compatibility)
- Qwen VL utilities
- AutoGen

⚠️ **Why separate environment?**
vLLM 0.11.0 has stricter PyTorch version requirements. The isolated environment prevents dependency conflicts.

---

## 🎓 Full Training Pipeline

> **⚠️ This section is for training models from scratch. If you only want to run inference with pre-trained models, see the [Inference Only section](#-inference-only-quick-start-) above.**

### **Segmentation Training** 🗺️

#### **Prerequisites**

**Launch Qwen vLLM Server** (required for geometry alignment):

```bash
# Activate Qwen environment
source .qvenv/bin/activate

# Start vLLM server on port 8000
bash segmentation/align_polys/run_qwen_vllm.sh
```

This starts a local OpenAI-compatible server with:

- Model: Qwen3-VL-30B-A3B-Thinking
- Tensor parallelism: 4-way (across 4 GPUs)
- Features: Chunked prefill, reasoning parser

#### **Run Training Pipeline**

```bash
# Activate main environment
source .venv/bin/activate

# Run full segmentation training
bash run_seg_training.sh
```

**Pipeline Steps**:

1. **Geometry Alignment** (~14 hours)
   - `align_geo_coords_to_px.py`: VLM aligns geographic → pixel coordinates
   - Output: `data/geom_px_df.csv`

2. **Dataset Creation** (~5 minutes)
   - `preprocessing/create_dataset.py`: Merge train/test, create K-folds
   - Output: `data/df.csv`

3. **Segmentation Model Training** (~1.5 hours)
   - `segmentation/train/train_seg.py`: Train Unet++ with EfficientNet-B5
   - Output: `outputs/seg_model_checkpoints/`

4. **Bias Model Training** (~10 minutes)
   - `segmentation/train/train_bias.py`: Learn surveyor-specific priors
   - Output: `outputs/bias_model_checkpoints/bias_model.pt`

---

### **Text Extraction Training** 📄

```bash
# Activate main environment
source .venv/bin/activate

# Run full text extraction training
bash run_text_training.sh
```

**Pipeline Steps**:

1. **Download Models**
   - `download_models.py`: Download Qwen3-VL-8B checkpoints

2. **Image Patchification**
   - `patchify_images.py`: Create 7 crops per image

3. **Label Correction**
   - `automated_label_correction.py`: VLM corrects training labels

4. **First Fine-Tune + Pseudo Labels** (~10 minutes)
   - `train_and_create_pseudo.py`: Fine-tune on corrected labels, generate pseudos

5. **Final Fine-Tune**
   - `train_with_pseudos.py`: Retrain on corrected + pseudo labels
   - Output: `outputs/qwen3vl8b_finetuned_lora/`

---

## 🏆 Competition Results

This solution achieved **1st place** on the [Barbados Lands and Surveys Plot Automation Challenge](https://zindi.africa/competitions/barbados-lands-and-surveys-plot-automation-challenge/leaderboard) on Zindi.

### **Final Scores**

| Metric | Public Leaderboard | Private Leaderboard |
|--------|-------------------|---------------------|
| **Overall Score** | **0.974424662** | **0.981903389** |
| Multi-Column Accuracy | 0.991780821 | 0.997260273 |
| IoU Polygon | 0.97757391 | 0.974757698 |
| Word Error Rate | 0.118965395 | 0.046535418 |

**Team**: Survey🤖 | **Submissions**: 117

### **Evaluation Metrics**

The competition used a weighted multi-metric evaluation:

- **IoU Polygon** (50%): Intersection over Union for land parcel shapes
- **Word Error Rate** (20%): Accuracy of extracted text (TargetSurvey field)
- **Multi-Column Accuracy** (30%): Exact match across 5 metadata fields

**Final Score** = `(0.5 × IoU) + (0.2 × (1 - WER)) + (0.3 × MCA)`

---

## 🔗 Key Technologies

| Component | Technologies |
|-----------|-------------|
| **Segmentation** | PyTorch Lightning, Segmentation Models PyTorch (SMP), Unet++, EfficientNet-B5, Albumentations |
| **Text Extraction** | Unsloth, Transformers, TRL, Qwen3-VL, BitsAndBytes (4-bit quantization) |
| **Geometry** | Shapely, GeoPandas, OpenCV, scikit-image, Douglas-Peucker smoothing |
| **VLM Serving** | vLLM, AutoGen, OpenAI-compatible API |
| **Training** | PyTorch 2.5.1, CUDA 12.4, Mixed Precision (AMP) |
| **Config** | YAML, python-box |

---

## 📧 Contact

**Author**: Darius Moruri

**Email**: [moruridarius@gmail.com](mailto:moruridarius@gmail.com)

For questions, collaboration opportunities, or inquiries about this solution, feel free to reach out via email.

---

## 🙏 Acknowledgments

- **Barbados Lands and Surveys Department** for providing the challenge data and real-world problem
- **Zindi Africa** for hosting the competition platform
- **Qwen Team** for developing excellent Vision-Language Models
- **Unsloth** for efficient LoRA fine-tuning tools that made text extraction training feasible

---

**🏝️ Built for the Barbados Lands and Surveys Department**

*Modernizing land records, one survey plan at a time.* ✨
