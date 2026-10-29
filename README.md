# MPI-CTMixer

Reference implementation of **MPI-CTMixer** (Multi-Pixel Inference CTMixer) from the paper
*"MPI-CTMixer: Efficient Multi-Pixel Inference for Hyperspectral Classification of Camouflaged Objects With In-Scene
and Scene-Separated Evaluation"* (IEEE Access, under revision).

Each 11 x 11 hyperspectral patch predicts its central **K x K** block (K = 9) in one forward pass. At test time the
class probabilities of overlapping output tiles (stride 5) are summed, which gives full-scene label maps with about
25x fewer patch evaluations than center-pixel inference. The backbone modifies CTMixer with a plain-conv progressive
spectral reduction (PSR) block and a dense (groups = 1) Q/K/V projection in LG-MHSA.

## What is in this repository

| | |
|---|---|
| `mpi_ctmixer/` | model, data pipeline, training/evaluation engine, overlapping multi-pixel inference |
| `scripts/train.py` | training on Indian Pines / Pavia University / YU Paint (paper recipe) |
| `scripts/evaluate.py` | test-pixel OA / AA / Kappa with non-overlapping inference (stride 9) and overlapping inference (stride 5) |
| `scripts/infer_one_image.py` | label map of **one** image (benchmark scene or your own cube) |
| `scripts/prepare_benchmarks.py` | converts the benchmark `.mat` files into the npz layout used here |
| `scripts/demo_concealhsi_samples.py` | smoke test on the released ConcealHSI sample patches, including the released ConcealHSI weights |
| `weights/` | MPI-CTMixer (K = 9) weights trained on Indian Pines, Pavia University and YU Paint (split 0), and on ConcealHSI under the pixel-pooled protocol |
| `data/splits/` | the five 10 % / 10 % / 80 % pixel splits (split seeds 0-4) of each benchmark used in the paper |
| `data/concealhsi_samples/` | two 11 x 11 x 150 patches for each of the 13 ConcealHSI classes (26 patches) |

**Not released:** the full ConcealHSI dataset (279 annotated scenes, hyperspectral cubes, annotations, scene
metadata and splits) and the ConcealHSI models of the scene-separated and patch-disjoint protocols. Distribution of the
defense-related concealment-scenario imagery is restricted under the data-management policy of the partner
organization. All benchmark experiments (Indian Pines, Pavia University, YU Paint) can be reproduced with this
repository. For ConcealHSI, the released pixel-pooled weights and the sample patches let you run the patch-based
multi-pixel model exactly as in the paper; the ConcealHSI accuracies cannot be recomputed, because the data are not
available.

## Install

```bash
pip install -r requirements.txt        # Python >= 3.9, PyTorch >= 2.0 (CPU is enough for evaluation / inference)
python tests/smoke_test.py             # or: python -m pytest tests
```

## Data

1. Download Indian Pines (corrected) and Pavia University from the
   [GIC Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
   page. These third-party scenes are not redistributed here.
2. Download `yu_paint_data2_gt2.zip` from the GitHub release of this repository (the in-house YU Paint scene) and unzip it.
3. Convert everything:

```bash
python scripts/prepare_benchmarks.py \
  --ip_data Indian_pines_corrected.mat --ip_gt Indian_pines_gt.mat \
  --pu_data PaviaU.mat --pu_gt PaviaU_gt.mat \
  --yu_data data2.mat --yu_gt gt2.mat
```

Indian Pines is stored zero-padded to 149 x 149; the provided split files index that layout.

## Reproduce the benchmark results (released weights)

```bash
python scripts/evaluate.py --dataset indian_pines
python scripts/evaluate.py --dataset pavia_university
python scripts/evaluate.py --dataset yu_paint
```

Expected output for the split-0 weights (K = 9, overlapping inference with stride 5, held-out 80 % test pixels, each
pixel scored once; these are single runs on split 0, the paper reports means over five different splits):

| dataset | OA | AA | Kappa |
|---|---|---|---|
| Indian Pines | 0.98661 | 0.98403 | 0.98473 |
| Pavia University | 0.99784 | 0.99614 | 0.99714 |
| YU Paint | 0.9989 | 0.9976 | 0.9984 |

(CPU and GPU floating-point summation can flip a handful of pixels in the last digit of YU Paint.)

## Predict one image

```bash
python scripts/infer_one_image.py --dataset yu_paint --out_dir outputs/yu_paint
python scripts/infer_one_image.py --cube my_scene.npy --weights my_weights.pth     # your own H x W x C cube
```

## Train from scratch

```bash
python scripts/train.py --dataset indian_pines --seed 0 --out_dir runs/ip_seed0     # 500 epochs
python scripts/evaluate.py --dataset indian_pines --weights runs/ip_seed0/mpi_ctmixer_k9_indian_pines_seed0.pth
```

Recipe: Adam (lr 5e-4, weight decay 1e-4), cosine schedule, batch 512, 500 epochs, sqrt-inverse class-weighted
cross-entropy, checkpoint with the best validation OA. The training loop reproduces the research code exactly
(identical losses and validation metrics over the first epochs on CPU), but GPU training is not bit-wise
deterministic, so expect run-to-run variation (the paper reports the mean and standard deviation over five different
train/validation/test splits). `--seed` selects both the random seed and the split file.

## ConcealHSI sample patches

`data/concealhsi_samples/concealhsi_samples.npz` holds two 11 x 11 x 150 patches for each of the 13 classes
(barracks, grass, tree, camouflage, friendly forces, enemy forces, tank, hair, skin, shadow, sky, building,
layperson), taken from different acquisitions. `patches` are channel-standardised and ready for the model,
`label_window` gives the class of every pixel in the window (-1 = unlabeled), `center_class` the class of the patch
center. `preview_rgb.png` shows an RGB reconstruction of every patch (bands averaged over 620-700 / 495-570 / 450-495 nm for
R / G / B, one common contrast stretch) next to its label window. `python scripts/demo_concealhsi_samples.py` runs a
forward pass with the K = 9 head, a few optimisation steps, plots the mean spectra, and predicts the samples with the
released ConcealHSI weights.

## ConcealHSI weights (pixel-pooled protocol)

`weights/mpi_ctmixer_k9_concealhsi_pixel_pooled_seed0.pth` is the MPI-CTMixer (K = 9) model of the paper's
pixel-pooled ConcealHSI experiment: labeled pixels of all 279 scenes pooled and split 10 % / 10 % / 80 % per class,
only training-split labels inside each 9 x 9 target block supervised, checkpoint with the best validation OA
(epoch 488). It expects 150 bands (470-900 nm) and predicts 13 classes (`class_names` in the file). Inputs are
standardised per channel with the dataset-wide statistics stored in the file (`norm_mean`, `norm_std`), and
`infer_one_image.py` applies them automatically:

```bash
python scripts/demo_concealhsi_samples.py                                   # the 26 released sample patches
python scripts/infer_one_image.py --cube my_cube.mat \
  --weights weights/mpi_ctmixer_k9_concealhsi_pixel_pooled_seed0.pth --out_dir outputs/mine   # an H x W x 150 cube
```

The sample patches are taken from the training data of this model, so its predictions on them (all 26 centre pixels
correct) check the pipeline, not the accuracy. Under this protocol, test pixels come from the training scenes, so the
weights reflect in-scene performance, not generalisation to unseen acquisitions (see the scene-separated results in
the paper).

## Origin

The backbone is derived from the official CTMixer implementation ([ZJier/CTMixer](https://github.com/ZJier/CTMixer),
Chen et al., *Convolution Transformer Mixer for Hyperspectral Image Classification*, IEEE GRSL 2022); see the header of
`mpi_ctmixer/model.py` for the modifications.
