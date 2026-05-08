# 3D Gaussian Splatting: Impact of Spatial Camera Distribution on Novel View Synthesis

## Team Members
* [Abdul Moeez Khurshid]
* [Muhammad Mustafa Javed]

## Abstract
This project conducts a rigorous empirical ablation study on the 3D Gaussian Splatting (3DGS) architecture. While 3DGS achieves state-of-the-art photorealism, it relies heavily on dense, 360-degree camera coverage initialized via COLMAP. Our research investigates the architecture's failure thresholds by isolating two specific data constraints: **Uniform Data Sparsity** (reducing frame count while preserving angular diversity) and **Spatial Bias** (simulating incomplete sweeps where cameras are clustered to a single hemisphere). 

We mathematically demonstrate that 3DGS is highly resilient to uniform data starvation but suffers catastrophic "geometry collapse" when subjected to spatial bias.

## Methodology & Experiments

Our study utilizes the `garden` scene from the Mip-NeRF 360 dataset. We designed a custom Python pipeline to deterministically subsample the dataset into six distinct conditions:

1. **100% Baseline:** Full 360-degree ground truth coverage.
2. **Density Ablation (Uniform):** 75%, 50%, and 25% data retention. Images were evenly spaced across the capture sequence to maintain full angular diversity.
3. **Spatial Bias (Clustered):** 50% data retention, physically restricted to a single hemisphere. 
   * *Algorithm:* We extracted world-space camera coordinates from the COLMAP `images.bin` file by converting the quaternion ($qvec$) and translation vector ($tvec$). We calculated the scene centroid and used directional vectors to cluster the top 50% spatial nearest-neighbors to the +X (Front) and -X (Back) axes.

## Results & Geometry Collapse

*Note: Models were trained on dual Nvidia T4 GPUs for 15,000 iterations.*

| Experimental Condition | PSNR | SSIM | LPIPS |
| :--- | :--- | :--- | :--- |
| **100% Baseline** | **27.03** | **0.8588** | **0.1177** |
| 75% Uniform | 26.45 | 0.8482 | 0.1249 |
| 50% Uniform | 25.74 | 0.8245 | 0.1372 |
| 25% Uniform | 22.43 | 0.7077 | 0.2067 |
| **50% Clustered Front** | **21.56** | **0.6819** | **0.2479** |
| **50% Clustered Back** | **21.65** | **0.6926** | **0.2503** |

### Key Findings
1. **Resilience to Sparsity:** The 3DGS architecture shows strong resilience to uniform frame drops. Reducing the dataset by 50% only resulted in a ~1.3 dB drop in PSNR.
2. **Proof of Geometry Collapse:** When the 50% Clustered Front model was evaluated exclusively on its own front-facing test set, it scored a highly respectable **26.23 PSNR**. However, to prove geometry collapse, we implemented a **Decoupled Global Evaluation**, forcing the biased model to render from the global 100% test camera poses. The PSNR violently dropped by ~5 dB down to **21.56**, and LPIPS error more than doubled. This mathematically quantifies the architectural breakdown in unobserved spatial regions.

## Technical Improvements & Pipeline Architecture

To execute this study within constrained cloud environments (Kaggle), we engineered several structural improvements to the standard 3DGS pipeline:

* **Decoupled Evaluation Pipeline:** Custom cross-evaluation scripts to prevent spatially biased models from artificially inflating metrics by "grading their own homework" on localized holdout sets.
* **VRAM Fragmentation Mitigation:** Integrated `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to prevent silent out-of-memory kernel crashes during the aggressive Gaussian densification phase.
* **Parallel Batched Training:** Engineered a synchronized subprocess queue to train models simultaneously across isolated dual T4 GPUs, optimizing compute time and bypassing Kaggle's 12-hour session limit.
* **Dataset Reader Patch:** Applied an automated C++ / Python file existence check to `scene/dataset_readers.py` to allow the rasterizer to gracefully handle symlinked subsampled datasets without crashing on missing files.
