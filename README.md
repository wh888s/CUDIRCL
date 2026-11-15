# CUDIRCL

**Enhancing diagnostic accuracy in multi-client medical imaging via proxy sharing in unified deep models**

PyTorch reference code for CUDIRCL. Each experiment is contained in one script and can be run through `main.py` or directly.

## Experiments

| Experiment | Script | Clients |
| --- | --- | --- |
| T1 CT reconstruction and T6 external-client adaptation | `T1_T6_ct_reconstruction.py` | 1–6 |
| T1 clinical CT denoising | `T1_ct_denoising.py` | 25–28 |
| T3 abdominal multi-organ CT segmentation | `T3_ct_segmentation.py` | 11–14, 3 |
| T2 MRI reconstruction | `T2_mri_reconstruction.py` | 7–10 |
| T4 prostate MRI segmentation | `T4_mri_segmentation.py` | 15–20 |
| T5 peripheral blood cell classification | `T5_pathology_classification.py` | 21–24 |
| T7 segmentation with and without proxy DP | `T7_privacy_protection.py` | 15–20 |

## Installation

Use Python 3.12. Install matching PyTorch and torchvision builds for your CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

CT segmentation uses [nnUNet v1](https://pypi.org/project/nnunet/1.7.1/); MRI reconstruction uses [fastMRI](https://pypi.org/project/fastmri/0.3.0/). CT reconstruction uses precomputed FBP inputs and a native differentiable backprojection layer.

## Data

| Task | Data type | Source |
| --- | --- | --- |
| T1 / T6 reconstruction | CT sinograms and images | NIH-AAPM-Mayo, Low-Dose CT Image and Projection Data, AbdomenCT-1K, CTSpine1K; two private hospital cohorts for T6 |
| T1 denoising | Real low-dose CT images | Four private hospital cohorts |
| T2 | T1-weighted brain MRI k-space/images | fastMRI, M4Raw, Calgary-Campinas-359, BraTS |
| T3 | Abdominal CT volumes and organ masks | CHAOS-CT, BTCV, WORD, AMOS, AbdomenCT-1K; annotations from AbdomenAtlas 1.0 |
| T4 / T7 | T2-weighted prostate MRI and masks | PROMISE12 (BIDMC/HK), I2CVB, NCI-ISBI 2013 (ISBI/ISBI15), MSD |
| T5 | RGB peripheral blood cell images | PBC dataset, Hospital Clinic of Barcelona |

## Usage

```bash
python main.py --help
python main.py T2_mri_reconstruction --help
python main.py T1_T6_ct_reconstruction train --data-root DATA --manifest split.csv --output runs/ct
python main.py T1_T6_ct_reconstruction test --data-root DATA --manifest split.csv --output runs/ct
```

Run from the repository folder using prepared data and split manifests. Select a script name without `.py`; use its `--help` for task-specific options. For T6, add `--task T6` and optionally `--initialize-from` with a T1 checkpoint.

## Outputs

Checkpoints are saved under `--output`. Testing exports predictions, probabilities where applicable, and necessary metadata. Evaluate these exports separately:

```bash
python evaluation.py T1 --results runs/ct/results --output metrics/ct
python evaluation.py T4 --results runs/prostate/results --references prostate.csv --data-root DATA --output metrics/prostate
python evaluation.py --help
```

`evaluation.py` covers reconstruction PSNR/SSIM/RMSE, segmentation DSC/ASSD/JC, classification ACC/ROC/AUC, clinical CT CNR, and supplied reader ratings. Ground-truth labels, CNR ROIs and reader scores are supplied externally when needed; `--help` lists their inputs.

## License

MIT License. See [LICENSE.txt](LICENSE.txt). Datasets and trained weights are not included.
