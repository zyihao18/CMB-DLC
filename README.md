<div align="center">

# CMB-DLC

**Cerebral Microbleed Detection, Localization, and Counting**

Research software for susceptibility-weighted imaging (SWI)

[📦 Quick start](#quick-start) · [🧠 Method](#method) · [🗂️ Datasets](#datasets) · [🏛️ Affiliations](#affiliations)

</div>

---

## Affiliations

- **Fudan University**, Shanghai, China
- **Department of Radiology, The Fifth People's Hospital of Shanghai, Fudan University**, Shanghai 200240, China

## Quick start

Use Python 3.10 or later on Linux or another POSIX system. The demo runs on synthetic scores without model access.

~~~sh
python -m pip install .
demo_dir=$(mktemp -d)
python examples/synthetic_scores.py "$demo_dir/scores.json"
cmb-dlc score --input "$demo_dir/scores.json" --threshold 0.5
~~~

The demo retains two findings. It counts one in the left frontal region and records the other as having an unresolved location. For real study inputs, see the [command guide](data/README.md).

## Method

| Stage | CLI | What it does |
| --- | --- | --- |
| Detection | `detect` | Screens the full SWI volume with a 2.5D U-Net using 32, 64, and 128 pixel crops and five slices per scale. Candidate components use a voxel cutoff of 0.3 and a size range of 2 to 500 voxels. |
| Visual assessment | `evidence` | Prepares images and prompts for seven vision-language model (VLM) assessments. Omits unavailable inputs and records the remaining tasks. |
| Score fusion | `score` | Combines the convolutional neural network (CNN) score (0.2) and the VLM score (0.8). Task weights are normalized over the available assessments. |
| Localization and counting | `register` → `localize` → `score` | Uses SynthStrip and ANTs to register the Microbleed Anatomical Rating Scale (MARS) atlas. The VLM reviews paired triplanar SWI and magnitude images with atlas overlays. Counts include retained findings with a resolved region and side. |

The manuscript reports a retention cutoff of **0.392399** for FDUCMB. Pass a cutoff explicitly when running `score`.

## Datasets

| Dataset or resource | Access |
| --- | --- |
| CUHK SWI microbleed dataset | [Official project page](https://www.cse.cuhk.edu.hk/~qdou/cmb-3dcnn/cmb-3dcnn.html) |
| VALDO microbleeds | [Official task page](https://valdo.grand-challenge.org/Task2/) · [Training data](https://zenodo.org/records/4687995) |
| FDUCMB | Controlled research access |
| MARS atlas | [Atlas and source code](https://github.com/v-sundaresan/microbleed-size-counting) · [Sundaresan et al. (2025)](https://doi.org/10.1186/s41747-024-00544-z) |

The [`data/` folder](data/) contains input templates. Keep scans, checkpoints, model responses, and generated evidence outside the repository. The CLI prepares evidence locally. Use de-identified images and an authorized provider for VLM requests.

## Verification

Run `python -m unittest discover -s tests`. The code uses the study's v4 detector. The synthetic demo checks response handling, score fusion, and regional counts. Study-level comparisons require the original cohort, model outputs, and evaluation settings.
