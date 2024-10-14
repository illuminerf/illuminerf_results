<h1 align="center">IllumiNeRF Benchmark Results</h1>

**IllumiNeRF: 3D Relighting without Inverse Rendering. NeurIPS 2024.**<br>
[Xiaoming Zhao](https://xiaoming-zhao.com/), [Pratul P. Srinivasan](https://pratulsrinivasan.github.io/), [Dor Verbin](https://dorverbin.github.io/), [Keunhong Park](https://keunhong.com/), [Ricardo Martin-Brualla](https://ricardomartinbrualla.com/), and [Philipp Henzler](https://henzler.github.io/). 

### [Project Page](https://illuminerf.github.io/) | [Paper](https://arxiv.org/abs/2406.06527)

---

This repo contains
1. [A comprehensive set of qualitative benchmark results produced by IllumiNeRF](https://github.com/illuminerf/illuminerf_results/releases), including results for [TensoIR](https://haian-jin.github.io/TensoIR/) and [Stanford-ORB](https://stanfordorb.github.io/) benchmarks;
2. Scripts for reproducing the quantitative metrics reported in the paper, based on these qualitative results.

While we are unable to release the full code, primarily because 1) our Relighting Diffusion Model (RDM) is built on a proprietary diffusion model, and 2) [UniSDF](https://fangjinhuawang.github.io/UniSDF/) has yet to release its code, we hope that these benchmark results will facilitate future comparisons with IllumiNeRF.

---

## Table of Contents

- [1 Environment Setup](#1-environment-setup)
- [2 Full Qualitative Results](#2-full-qualitative-results)
- [3 Quantitative: Stanford-ORB Benchmark](#3-quantitative-stanford-orb-benchmark)
- [4 Quantitative: TensoIR Benchmark](#4-quantitative-tensoir-benchmark)
  - [4.1 Correctness of Our Script](#41-correctness-of-our-script)
  - [4.2 Results of IllumiNeRF](#42-results-of-illuminerf)
- [Citation](#citation)

## 1 Environment Setup

This code has been tested on Ubuntu 20.04 with CUDA 11.8 on NVIDIA RTX A6000 GPU (driver 470.42.01).

We recommend using `conda` for virtual environment control and [`libmamba`](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for a faster dependency check.

```bash
# setup libmamba
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# create virtual environment
conda env create -f environment.yaml
```

Throughout this README, we define the following environment variable for the illustration purpose:
```bash
cd /path/to/this/repo
export ILLUMINERF_RES_ROOT=$PWD
```

## 2 Full Qualitative Results

Please download full quanlitative results from the [release page](https://github.com/illuminerf/illuminerf_results/releases/):

```bash
# download results
ALL_RES_FNAMES=(illuminerf_stanford_orb_mult_gpu illuminerf_stanford_orb_single_gpu illuminerf_tensoir_mult_gpu illuminerf_tensoir_single_gpu)

# download
printf "%s\0" "${ALL_RES_FNAMES[@]}" | xargs -0 -n 1 -I {} -P 4 wget https://github.com/illuminerf/illuminerf_results/releases/download/v0.1/{}.zip -P ${ILLUMINERF_RES_ROOT}/data/
# unzip
printf "%s\0" "${ALL_RES_FNAMES[@]}" | xargs -0 -n 1 -I {} -P 4 unzip ${ILLUMINERF_RES_ROOT}/data/{}.zip -d ${ILLUMINERF_RES_ROOT}/data/
```

After running the above command, you should have a structure as the following:

```
.
+-- data
|  +-- illuminerf_stanford_orb_mult_gpu
|  |  +-- benchmark
|  |  ...
|  +-- illuminerf_stanford_orb_single_gpu
|  |  +-- benchmark
|  |  ...
|  +-- illuminerf_tensoir_mult_gpu
|  |  +-- armadillo
|  |  ...
|  +-- illuminerf_tensoir_single_gpu
|  |  +-- armadillo
|  |  ...
```

## 3 Quantitative: Stanford-ORB Benchmark

Please refer to [this issue](https://github.com/StanfordORB/Stanford-ORB/issues/10) on how on reproduce the quantitative results on the Stanford-ORB benchmark.

## 4 Quantitative: TensoIR Benchmark

Our reported metrics were initially calculated using internal tools. To ensure reproducibility, we provide [a standalone script](./compute_metrics_tensoir.py) that can be used independently.

To use that script, please download the benchmark data from [the official TensoIR repo](https://github.com/Haian-Jin/TensoIR?tab=readme-ov-file#downloading) and place them under `${ILLUMINERF_RES_ROOT}/data/TensoIR_Synthetic` as:
```
.
+-- data
|  +-- TensoIR_Synthetic
|  |  +-- armadillo
|  |  +-- ficus
|  |  +-- hotdog
|  |  +-- lego
```

### 4.1 Correctness of Our Script

If you want to verify the correctness of our script, please download the [official TensoIR results](https://github.com/Haian-Jin/TensoIR?tab=readme-ov-file#for-pre-trained-checkpoints-and-results-please-see) from the official repo and place them under `${ILLUMINERF_RES_ROOT}/data/TensoIR_Results` as the following
```
.
+-- data
|  +-- TensoIR_Results
|  |  +-- armadillo_single
|  |  +-- ficus_single
|  |  +-- hotdog_single
|  |  +-- lego_single
```

Then run the following command:
```bash
conda activate illuminerf_results && \
export PYTHONPATH=${ILLUMINERF_RES_ROOT}:$PYTHONPATH && \
python ${ILLUMINERF_RES_ROOT}/compute_metrics_tensoir.py \
  --result_type tensoir \
  --results_path ${ILLUMINERF_RES_ROOT}/data/TensoIR_Results \
  --tensoir_data_path ${ILLUMINERF_RES_ROOT}/data/TensoIR_Synthetic/ \
  --result_already_rescaled \
  --nproc 10
```
The quantitative results will be saved in `${ILLUMINERF_RES_ROOT}/data/TensoIR_Results/metrics.json` and you could observe the following comparison:

|  | PSNR | SSIM | LPIPS-VGG |
| --- | ---: | ---: | ---: |
| From [TensoIR paper](https://arxiv.org/abs/2304.12461)'s Tab. 1 | 28.580 | 0.944 | 0.081 |
| From the script | 28.514 | 0.944 | 0.081 |


### 4.2 Results of IllumiNeRF

Run the following command to reproduce quantitative results reported in the paper:

```bash
# - for "--results_path", we can choose from
#   - illuminerf_tensoir_mult_gpu
#   - illuminerf_tensoir_single_gpu
# - since we provide both raw as well as already-rescaled renderings from IllumiNeRF:
#   - use "--no-result_already_rescaled" to compute from raw renderings
#   - use "--result_already_rescaled" to compute from already-rescaled renderings
conda activate illuminerf_results && \
export PYTHONPATH=${ILLUMINERF_RES_ROOT}:$PYTHONPATH && \
python ${ILLUMINERF_RES_ROOT}/compute_metrics_tensoir.py \
  --result_type illuminerf \
  --results_path ${ILLUMINERF_RES_ROOT}/data/illuminerf_tensoir_mult_gpu \
  --tensoir_data_path ${ILLUMINERF_RES_ROOT}/data/TensoIR_Synthetic/ \
  --no-result_already_rescaled \
  --nproc 10
```

The quantitative results will be saved in `${ILLUMINERF_RES_ROOT}/data/illuminerf_tensoir_mult_gpu/metrics_from_{raw, rescaled}.json` and you could observe the following comparison:

|  | PSNR | SSIM | LPIPS-VGG |
| --- | ---: | ---: | ---: |
| **[mult-GPU]** From paper's Tab. 1 | 29.709 | 0.947 | 0.072 |
| **[mult-GPU]** From the script | 29.704 | 0.947 | 0.072 |
| **[single-GPU]** From paper's Tab. 1 | 29.245 | 0.946 | 0.073 |
| **[single-GPU]** From the script | 29.240 | 0.945 | 0.073 |


## Citation
>Xiaoming Zhao, Pratul P. Srinivasan, Dor Verbin, Keunhong Park, Ricardo Martin Brualla, and Philipp Henzler. IllumiNeRF: 3D Relighting without Inverse Rendering. NeurIPS 2024.
```
@inproceedings{zhao2024illuminerf,
    author    = {Xiaoming Zhao and Pratul P. Srinivasan and Dor Verbin and Keunhong Park and Ricardo Martin Brualla and Philipp Henzler},
    title     = {{IllumiNeRF: 3D Relighting without Inverse Rendering}},
    booktitle = {NeruIPS},
    year      = {2024},
}
```