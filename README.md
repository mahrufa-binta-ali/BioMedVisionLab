---
title: BioMedVisionLab
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# BioMedVisionLab

**Biomedical Imaging Visualization Toolkit for Retrieval and Super-Resolution**

BioMedVisionLab is a polished, professor-facing Streamlit workbench for evaluating biomedical image-like data. It supports visual inspection, embedding comparison, retrieval benchmarking, super-resolution baseline evaluation, future model-output comparison, contact-map visualization, and CSV experiment export.

1. Chest X-ray similarity retrieval visualization
2. Biomedical image super-resolution demo
3. Grant Alignment Lab for chemical imaging and Hi-C contact-map research directions

BioMedVisionLab is upload-first for deployed demonstrations. The app can use a built-in/local CXR embedding index when one is available, but it also works without a local Kaggle dataset by letting users upload biomedical images or contact-map matrices directly in the browser.

The project is intended for education, research prototyping, professor-facing demonstrations, and portfolio presentation. It is not intended for medical diagnosis, triage, treatment, or clinical decision-making.

## Overview

BioMedVisionLab combines representation learning, medical image retrieval, CXR-specific encoder support, batch retrieval quality measurement, super-resolution baseline evaluation, model-output comparison, Hi-C-like matrix visualization, and report export in one interactive dashboard.

- `build_index.py` scans a local chest X-ray dataset, extracts embeddings, and saves searchable encoder-specific indexes.
- `app.py` launches a Streamlit toolkit with tabs for retrieval, super-resolution visualization, and grant-alignment demos.

## Motivation

Biomedical imaging research often needs visual, explainable tools for understanding what models learn and how image quality affects downstream interpretation. This project focuses on showing model behavior and image-processing concepts without making diagnostic claims.

## Module 1: CXR Retrieval

The CXR retrieval module visualizes similarity search over a local chest X-ray dataset.

Features:

- Loads saved embeddings from `embeddings/`
- Shows a clear **Local CXR index available** or **Upload-only mode** status
- Supports dataset image queries and uploaded image queries
- If no saved embeddings are available, supports upload-first retrieval:
  - upload a query image
  - upload a small reference gallery of JPG/PNG/JPEG files
  - build temporary in-session embeddings for the uploaded gallery
  - retrieve visually similar uploaded reference images
- Uploaded gallery retrieval is temporary and is not saved
- Uses pretrained torchvision ResNet18 as a lightweight generic ImageNet feature-extraction baseline
- Adds pretrained torchvision ResNet50 as a deeper generic ImageNet feature-extraction baseline
- Optionally supports TorchXRayVision DenseNet121 as a CXR-specific embedding encoder when `torchxrayvision` is installed
- Uses 512-dimensional L2-normalized embeddings for ResNet18, 2048-dimensional L2-normalized embeddings for ResNet50, and DenseNet feature embeddings for TorchXRayVision
- Stores encoder-specific indexes in `embeddings/resnet18_imagenet/`, `embeddings/resnet50_imagenet/`, and `embeddings/torchxrayvision_densenet121/`
- Stores metadata for encoder name, embedding dimension, dataset size, build time, and device
- Lets the app sidebar switch between the generic ImageNet baseline and the CXR-specific encoder when their indexes are available
- Retrieves top-k visually similar images with cosine similarity
- Avoids returning the same image as the top result for dataset queries
- Shows query image, query filename, query label, image resolution, embedding dimension, and query source
- For uploaded CXR-index queries, runs a transparent input-domain check that looks for CXR-like grayscale, aspect ratio, and contrast properties
- Flags likely out-of-domain uploads, such as posters or banners, before interpreting CXR nearest-neighbor results
- Uses colored clinical-style labels for `NORMAL`, `PNEUMONIA`, agreement state, and retrieval confidence
- Includes a compact model card with encoder, embedding dimension, similarity metric, dataset, and clinical status
- Adds a clinical-style retrieval triage panel for query source, majority retrieved pattern, agreement, similarity range, and review status
- Reports retrieval quality metrics for dataset-selected queries:
  - Top-k label agreement count
  - Precision@K
  - Mean cosine similarity
- Shows Top-k sensitivity for Top-3, Top-5, and Top-10 label agreement when the query label is known
- Shows a retrieval confidence badge using a simple heuristic based on top similarity and, when available, Precision@K
- Makes clear that retrieval confidence is not clinical or diagnostic confidence
- Uses safer uploaded-query wording such as **Top retrieved dataset label** and states that retrieved labels are nearest-neighbor metadata, not predicted labels or diagnoses
- For out-of-domain uploads, marks retrieval confidence as **Out-of-domain / Low**, hides label agreement metrics as unavailable, and shows NORMAL/PNEUMONIA nearest labels only as debugging context
- Recommends custom uploaded-gallery retrieval for non-CXR images instead of using the CXR index
- Renames retrieved examples as a **Similar Case Board**
- Shows rank, filename, label badge, cosine similarity, query-label agreement, and a progress bar on each result card
- Includes a compact result table with rank, filename, label, cosine similarity, and label agreement
- Keeps compact charts for rank-based similarity scores and retrieved-label distribution
- Adds a query-vs-retrieved comparison view with query image, selected retrieved image, and a simple absolute difference map
- Adds an **Intensity Heatmap Preview** for the query image
- Exports the current retrieval experiment as CSV
- Adds batch retrieval evaluation for 20, 50, or 100 sampled queries
- Reports Mean Precision@K, label-specific Precision@K, Top-1 match rate, same-label counts, and retrieved-label distributions
- Exports a batch retrieval benchmark CSV with one row per retrieved result for comparing encoder experiments
- Adds an Encoder Comparison Benchmark when both indexes exist, using the same sampled queries to compare Mean Precision@K, class Precision@K, and Top-1 match rate
- Includes the Research Extension Roadmap in the CXR Retrieval tab only, so retrieval users can explore future encoder, explainability, text, microscopy, diffusion, and genomics directions without repeating the roadmap across every tab

The intensity heatmap is a simple grayscale intensity visualization. It is not Grad-CAM and does not provide model-based lesion localization.
The difference map is a visual comparison aid, not a clinical finding map.

## Module 2: Super-Resolution Demo

The super-resolution module demonstrates the basic image degradation and reconstruction problem.

Features:

- Select an image from the indexed CXR dataset or upload a biomedical image
- Works in upload-only deployments when no local CXR index exists
- Simulate low resolution by downsampling by `2x` or `4x`
- Reconstruct the image back to original size with bicubic interpolation
- Display original, simulated low-resolution, and bicubic enhanced images side by side
- Calculate MSE, PSNR, and SSIM when `scikit-image` is available
- Compare zoomed center crops of the original, low-resolution, and bicubic enhanced images
- Use a crop-size slider for quick local inspection
- Includes a baseline model card for degradation, reconstruction, metrics, and clinical status
- Adds an artifact risk card explaining that bicubic interpolation can smooth fine structures and cannot recover true missing anatomical detail
- Provides a large preview selector for original, simulated low-resolution, and bicubic enhanced images
- Supports uploading a model-enhanced image for comparison against the bicubic baseline
- Shows bicubic vs original and uploaded-output vs original metrics in a comparison table
- Adds batch super-resolution evaluation for 20, 50, or 100 indexed images
- Exports batch super-resolution benchmark CSV files for method comparison
- Exports the current super-resolution experiment as CSV
- Explains the relevance of throughput/resolution tradeoffs, microscopy/pathology, and future diffusion-based enhancement
- Adds a **Super-Resolution Exploration Studio** with prompts for uploaded learned outputs, crop-level inspection, metric deltas, and artifact checks

This module is a lightweight educational prototype. Bicubic super-resolution is an educational baseline, not a trained AI model or a clinical enhancement method.

## Module 3: Grant Alignment Lab

The Grant Alignment Lab frames BioMedVisionLab as a reusable visual evaluation workbench for biomedical imaging and multimodal biology research.

Features:

- Chemical Imaging Super-Resolution alignment card:
  - low-resolution biomedical/chemical imaging challenge
  - current support for degradation simulation, bicubic baseline, PSNR, SSIM, MSE, crop inspection, and artifact warnings
  - future diffusion-based super-resolution, microscopy/pathology datasets, and SRS/Raman-style image support
- Hi-C Contact Map Visualization alignment card:
  - sparse, noisy, high-dimensional image-like contact matrices
  - current support for matrix visualization and resolution-enhancement concepts
  - future real Hi-C/Micro-C upload, contact-map retrieval, and RNA-seq/ATAC-seq metadata integration
- Synthetic Hi-C contact map mini-demo:
  - generates synthetic contact matrices with diagonal structure and loop-like bright spots
  - simulates low-resolution contact maps and bicubic upscaling
  - displays original, low-resolution, upscaled, and absolute difference maps
  - reports Matrix MSE, Matrix PSNR, and optional Matrix SSIM
  - exports the synthetic Hi-C demo report as CSV
- Upload support for `.npy` or `.csv` square contact matrices:
  - validates square matrices
  - safely normalizes values
  - displays uploaded contact map, simulated low-resolution map, bicubic-upscaled map, and difference map
  - exports Hi-C matrix evaluation metrics as CSV
- Adds a **Multimodal Exploration Hub** for matrix uploads, low/high-resolution matrix comparison, diagonal profiles, and future metadata links

The deployed app can therefore demonstrate radiology image retrieval, biomedical super-resolution baselines, and contact-map matrix visualization without requiring users to download the repository or install the original local dataset.

Synthetic contact maps are used only to demonstrate the visualization interface. They are not biological results.
Uploaded contact maps are visualized as matrix images. This prototype does not yet perform biological feature calling.

## Biomedical Imaging Relevance

BioMedVisionLab connects to common biomedical imaging research workflows:

- Chest X-ray dataset exploration
- Visual search and content-based medical image retrieval
- Representation learning with pretrained vision models
- Retrieval confidence and uncertainty-style summaries for visual search behavior
- Clinical-style retrieval triage and Top-k sensitivity checks for research review
- Comparing generic ImageNet embeddings with CXR-specific embeddings
- Batch measurement of retrieval quality across repeated query samples
- Query-vs-retrieved visual comparison with a non-clinical difference map
- Explainable AI previews and future Grad-CAM visual grounding
- Image degradation and reconstruction experiments
- Super-resolution metrics for biomedical imaging baselines
- Artifact-risk framing for educational super-resolution baselines
- Comparing future learned or diffusion-based super-resolution outputs against bicubic interpolation
- Chemical imaging super-resolution grant prototypes
- Chromatin contact-map visualization and retrieval concepts
- Multimodal biology workbench patterns for image-like matrices and metadata
- Educational demonstrations for biomedical AI portfolios
- Conceptual links to microscopy and pathology super-resolution

## Connection to Core Concepts

**Representation learning:** ResNet18, ResNet50, or TorchXRayVision converts images into numerical embeddings that summarize visual patterns.

**Image retrieval:** Cosine similarity compares embeddings and ranks images by visual resemblance.

**Explainable AI:** The dashboard includes a simple intensity heatmap preview as a non-diagnostic visualization placeholder. Future work can add Grad-CAM or other model-based visual grounding.

**Super-resolution:** The demo degrades an image and reconstructs it, illustrating why high-resolution structure matters in biomedical imaging.

## Dataset Instructions

Place a small public chest X-ray dataset inside:

```text
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
```

The app also works if only `train/` and `test/` exist.

Supported image formats:

- `.jpg`
- `.jpeg`
- `.png`

## Setup Commands

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the CXR retrieval index with the default generic ResNet18 ImageNet encoder:

```bash
python build_index.py --encoder resnet18_imagenet
```

Build an optional TorchXRayVision DenseNet121 CXR-specific index:

```bash
python build_index.py --encoder torchxrayvision_densenet121
```

Indexes are stored separately:

```text
embeddings/
    resnet18_imagenet/
    resnet50_imagenet/
    torchxrayvision_densenet121/
```

Build the deeper generic ResNet50 ImageNet encoder:

```bash
python build_index.py --encoder resnet50_imagenet
```

ResNet18 is a lightweight generic natural-image baseline pretrained on ImageNet. ResNet50 is a deeper generic ImageNet baseline with a 2048-dimensional penultimate feature vector. TorchXRayVision DenseNet121 is pretrained for chest X-ray tasks and is included to support research comparisons between generic and CXR-specific visual representations.

Run the dashboard:

```bash
python -m streamlit run app.py
```

If `streamlit` is available on your PATH, this also works:

```bash
streamlit run app.py
```

## Deployment on Hugging Face Spaces

BioMedVisionLab is prepared for Hugging Face Spaces as a Streamlit app. The main entry file is:

```text
app.py
```

The Streamlit configuration is stored in:

```text
.streamlit/config.toml
```

with a `500 MB` upload limit and browser usage-stat collection disabled.

### Run Locally

Install dependencies and start the app:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Local CXR indexes are optional. If no embeddings are present, the app opens in upload-only mode for query/gallery retrieval, image super-resolution, and contact-map visualization.

### Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Select **Streamlit** as the Space SDK.
3. Upload or push the repository files, including:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `README.md`
4. Spaces will install dependencies from `requirements.txt` and run the Streamlit app from `app.py`.
5. Optional local embedding folders can be added under `embeddings/`, but deployed demos can also run without them through upload-first workflows.

### Deployment Limitations

- Uploaded files are temporary and are not saved as a persistent dataset.
- Large datasets should be uploaded as smaller representative samples.
- The app is not for diagnosis, triage, treatment, or clinical decision-making.
- Heavy diffusion-model training or large-scale model inference is not supported in this demo.
- If TorchXRayVision is unavailable, uploaded-image embedding falls back to ResNet18 and the app shows a warning.
- If `cooler` is unavailable, `.cool` and `.mcool` upload is disabled with a helpful message; `.csv` and `.npy` contact matrices still work.

## Method Details

For CXR retrieval, each image is:

1. Converted to RGB
2. Resized to `224x224`
3. Normalized with ImageNet mean and standard deviation
4. Passed through pretrained ResNet18 with the classifier removed
5. Converted into a 512-dimensional embedding
6. L2-normalized for cosine similarity search
7. Compared with indexed embeddings using cosine similarity
8. Summarized with rank-based similarity, label agreement, Precision@K, and mean similarity
9. Reviewed with model-card metadata, retrieval triage, Top-k sensitivity, and optional visual comparison
10. Exported as a CSV experiment report

Batch retrieval evaluation samples labeled dataset queries, retrieves Top-k neighbors while excluding the query image, and exports one row per retrieved result. This makes CSV exports useful for comparing generic and CXR-specific encoders.

Retrieval confidence is a heuristic:

- High confidence for dataset queries if top similarity is at least `0.90` and Precision@K is at least `0.80`
- Medium confidence if top similarity is at least `0.80`
- Low confidence otherwise
- Uploaded images use top similarity only because query-label agreement is unavailable

Uploaded CXR-index queries also run a simple out-of-domain input check before retrieval summaries are shown. The check uses transparent image heuristics only:

- whether the image is mostly grayscale
- whether the aspect ratio is roughly radiograph-like
- whether the image is highly colorful
- whether grayscale contrast is sufficient for a medical-image-like input

This check is not a clinical classifier. If an uploaded image appears out of domain, the app still allows nearest-neighbor retrieval but warns that results may be meaningless and displays nearest CXR dataset labels only as debugging metadata.

For super-resolution, each image is:

1. Converted safely to RGB
2. Downsampled by `2x` or `4x`
3. Reconstructed to original size using bicubic interpolation
4. Compared against the original image using MSE, PSNR, and optional SSIM
5. Displayed with center-crop comparisons for local detail inspection
6. Reviewed with artifact-risk and baseline-method cards
7. Exported as a CSV experiment report

ChemDiffuse-style comparison allows an externally generated enhanced image to be uploaded and compared against the original image and bicubic baseline using PSNR, SSIM, and MSE.

For the synthetic Hi-C mini-demo:

1. A synthetic square contact matrix is generated with diagonal structure and loop-like bright spots.
2. Noise is added using the dashboard slider.
3. The matrix is downsampled by `2x` or `4x`.
4. The low-resolution matrix is bicubic-upscaled to the original matrix size.
5. Matrix MSE, Matrix PSNR, and optional Matrix SSIM are reported.
6. The demo can be exported as a CSV report.

## Limitations

- ResNet18 and ResNet50 are pretrained on natural images, not chest X-rays.
- TorchXRayVision support requires optional dependency installation and a separately built index.
- Retrieval results show visual similarity, not clinical equivalence.
- The CXR retrieval module is intended for chest X-ray-like inputs. Out-of-domain images may still produce nearest-neighbor results, but these are not meaningful and are flagged by the app.
- Retrieval confidence is not clinical confidence.
- Retrieval triage is a research dashboard summary, not clinical triage.
- Query-vs-retrieved difference maps are visual comparison aids, not finding maps.
- Folder labels are used only for visualization.
- The intensity heatmap is not Grad-CAM and does not identify pathology.
- Bicubic interpolation is a simple baseline, not a learned super-resolution model.
- Bicubic interpolation may smooth fine detail and cannot recover true missing anatomical structure.
- Synthetic Hi-C contact maps are interface demonstrations only and are not biological results.
- Uploaded Hi-C/contact matrices are visualized as matrix images and are not biologically interpreted.
- PSNR is a basic image-quality metric and does not measure clinical usefulness.
- SSIM is an image-similarity metric and does not measure clinical usefulness.
- The project should not be used for diagnosis, triage, treatment, or clinical decision-making.

## CXR Retrieval Research Extension Roadmap

The Research Extension Roadmap is shown only in the CXR Retrieval tab. The Super-Resolution and Grant Alignment tabs have their own exploration sections instead of repeating the same roadmap.

- CXR-specific encoders such as TorchXRayVision or medical foundation models
- Grad-CAM visual grounding
- Report-aware retrieval using radiology text
- Microscopy/pathology super-resolution datasets
- Diffusion-based biomedical image enhancement
- Learned super-resolution models such as SRCNN, EDSR, or SwinIR variants
- Metadata-aware retrieval filters
- Hi-C contact map visualization as a future biomedical data visualization extension
- Real Hi-C/Micro-C upload, contact-map retrieval, and omics metadata integration
- Chemical imaging super-resolution with SRS/Raman-style data
- Grant-ready experiment report templates

## Disclaimer

BioMedVisionLab is a research visualization prototype and is not intended for medical diagnosis or clinical decision-making.

- Retrieval confidence is not diagnostic or clinical confidence.
- The intensity heatmap preview is not Grad-CAM.
- Bicubic super-resolution is an educational baseline, not a trained AI model.
- Synthetic Hi-C contact maps are not biological results.

## Suggested LinkedIn Project Description

Built **BioMedVisionLab**, a reusable biomedical visual evaluation workbench in Python and Streamlit. The app includes chest X-ray similarity retrieval using pretrained ResNet18, ResNet50, and optional TorchXRayVision embeddings with cosine similarity, retrieval triage, Top-k sensitivity, query-vs-result comparison, experiment report export, an educational super-resolution demo with MSE, PSNR, SSIM, artifact-risk framing, and a Grant Alignment Lab with a synthetic Hi-C contact-map mini-demo. The project highlights medical image retrieval, chemical imaging super-resolution, explainable AI, and chromatin contact-map visualization concepts for research and portfolio demonstration, not clinical diagnosis or biological interpretation.
