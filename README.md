# Learning Directly from Synthetic Point Clouds for "In-the-wild" 3D Face Recognition

Point clouds-based networks have achieved great attention in 3D object classification, segmentation, and indoor scene semantic parsing, but its application to 3D face recognition is still underdeveloped owing to two main reasons: lack of large-scale 3D facial data and absence of deep neural network that can directly extract discriminative face representations from point clouds. To address these two problems, a PointNet++ based network is proposed in this paper to extract face features directly from point clouds facial scans and a statistical 3D Morphable Model based 3D face synthesizing strategy is established to generate large-scale unreal facial scans to train the proposed network from scratch. A curvature-aware point sampling technique is proposed to hierarchically down-sample feature-sensitive points which are crucial to pass and aggregate discriminative facial features deeply. In addition, a novel 3D face transfer learning method is proposed to ease the domain discrepancy between synthetic and 'in-the-wild' faces.
Experimental results on two public 3D face benchmarks show that the network trained only on synthesized data can also be well generalized to 'in-the-wild' 3D face recognition. Our method achieves the state-of-the-art results by achieving an overall rank-1 identification rate of 99.46\% and 99.65\% on FRGCv2 and Bosphorus, respectively. Further, we evaluate on a self-collected dataset to demonstrate the robustness and application potential of our method.

## Data prepare
The original pipeline used MATLAB (`Data/GenerateTrainData.m`) and PCL. It is now
replaced by a pure-Python generator (see [Training on generated faces](#training-on-generated-faces)):
1. Download the [BFM2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html) h5 file `model2017-1_face12_nomouth.h5` and move it into `Data/`.
2. Run `python -m pointnet2.data.generate_faces --bfm Data/model2017-1_face12_nomouth.h5 --out Data/synthetic`.
   It computes normals and curvature itself, so PCL is not needed.

## Network train
### setup
Our work is based on [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch). We modify the sampling and network struture in our project.

The code no longer depends on `etw_pytorch_utils`; the few helpers it provided
(`SharedMLP`, `Seq`, checkpoint loading) now live in `pointnet2/utils/pytorch_utils.py`
and keep the same parameter names, so existing checkpoints (e.g.
`checkpoints/20191028_1000cls_model_best.pth.tar`) load unchanged.

Requirements: Python >= 3.8, an NVIDIA GPU, PyTorch with CUDA, and a CUDA toolkit
(`nvcc`) whose version matches the one PyTorch was built with (`python -c "import torch; print(torch.version.cuda)"`).

1. Install PyTorch for your CUDA version (see https://pytorch.org/get-started/locally/), then the rest:
   ```
   pip install -r requirements.txt
   ```
2. Build the CUDA ops (`pointnet2._ext`: furthest point sampling, ball query, grouping, interpolation):
   ```
   python setup.py build_ext --inplace
   ```
   This creates `pointnet2/_ext.*.so`. Alternatively install the package with
   `pip install -e . --no-build-isolation`.
   If you build on a machine without a visible GPU (e.g. in Docker), set the target
   architectures first, e.g. `export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9"`.
3. Check the build:
   ```
   python -c "import pointnet2._ext as e; print([n for n in dir(e) if not n.startswith('_')])"
   ```

Troubleshooting the build:
* `CUDA compiler and CUDA toolkit headers are incompatible` / `Unsupported .version` (ptxas):
  your `nvcc` and CUDA headers come from different CUDA releases. Use one toolkit whose
  version matches `torch.version.cuda`, and point `CUDA_HOME` at it.
* `cannot find -lcudart` with a pip-installed CUDA toolkit (`nvidia-cuda-*` wheels): only
  `libcudart.so.<major>` is shipped. Create a `libcudart.so` symlink to it in some folder
  and add that folder to `LIBRARY_PATH`.

Run the training scripts from the repository root (`python -m pointnet2.train...`) so
that the `pointnet2` package is importable.
### Example training
Training examples are provided by `pointnet2/train/train_cls.py` and `pointnet2/train/train_triplet.py`. `train_cls` pre-trains on generated data, and `train_triplet` fine-tunes on real faces for better results.
```
python -m pointnet2.train.train_cls -data_root Data/synthetic -checkpoint_dir runs/synthetic
python -m pointnet2.train.train_triplet -data_root /path/to/real_npy -model_checkpoint runs/synthetic/pointnet2_model_best.pth.tar
```

### Training on generated faces
One command generates the synthetic faces and pre-trains the network on them
(run from the repository root, after the setup above and downloading the BFM file):
```
bash scripts/train_synthetic.sh
```
Settings are environment variables, for example:
```
SUBJECTS=2000 SCANS=30 EPOCHS=15 WORKERS=16 bash scripts/train_synthetic.sh
```
| variable | default | meaning |
|---|---|---|
| `BFM` | `Data/model2017-1_face12_nomouth.h5` | BFM2017 model |
| `OUT` | `Data/synthetic` | where generated faces go |
| `RUN` | `runs/synthetic` | checkpoints and logs |
| `SUBJECTS` / `SCANS` | 1000 / 20 | identities (classes) and scans per identity |
| `POINTS` | 4096 | points per scan |
| `NOISE` | 0 | Gaussian vertex noise in mm (simulates a scanner) |
| `EPOCHS`, `BATCH`, `LR`, `CLASSIFIER` | 10, 32, 1e-3, AL | classification training (`AL`, `MCP` or `L`) |
| `REAL_DATA` | unset | folder of real `.npy` scans: also runs triplet fine-tuning |

Extra arguments go to `train_cls`, e.g. `bash scripts/train_synthetic.sh -resume`.
Both stages can be re-run safely: subjects that are already generated are skipped,
and `-resume` continues training from the last epoch.

The pipeline runs two steps you can also run on their own:

1. **Generation** (`python -m pointnet2.data.generate_faces --help`): for each subject, one
   random BFM shape; for each scan, a random expression (`--neutral_first` makes scan 000
   neutral). Per-vertex normals come from the mesh, and curvature is the PCL surface
   variation `l0/(l0+l1+l2)` over `--curvature_k` neighbours. Then `--num_points`
   vertices are sampled. Output is `OUT/<subject>/<subject>_<scan>.npy`, `[4096, 7]`,
   the same format as the real-data loader. Coordinates default to zero-mean, unit-sphere
   (`--normalize unit_sphere`) to match the real scans used for fine-tuning.
   `--normalize nose_mm100` gives the paper's nose-centred mm/100 instead. A few
   `.ply` previews are written to `OUT/_preview` so you can check the faces.
   Seeds are fixed per subject, and `--start_subject` adds more subjects to an existing set.
2. **Classification pre-training** (`python -m pointnet2.train.train_cls --help`):
   each subject is one class, and `-val_ratio` (default 0.1) of each subject's scans is
   held out to measure accuracy. The backbone is saved to
   `RUN/pointnet2_model_best.pth.tar`, which is what `train_triplet -model_checkpoint` takes.

### Triplet training on preprocessed `.npy` point clouds
`train_triplet` reads a folder (searched recursively) of preprocessed scans:

* file name `{subject}_{image_id}.npy` (the subject is the text before the first `_`)
* array shape `[4096, 7]`: XYZ (0-2, zero-mean, unit-sphere scaled), normals (3-5), curvature (6)

Because the clouds are already cleaned, centred, scaled and sampled to 4096 points, no
coordinate normalisation is applied. Subjects with fewer than `-min_samples` (default 20)
scans are dropped, and a per-subject split (`-val_ratio`, default 0.2) is held out for
validation (one gallery scan per subject, the rest are probes) unless `-val_root` is given.
Training uses **P x K batches with online batch-hard mining**: each batch holds P subjects
with K scans each (sampled with replacement when a subject has fewer than K), the whole
batch goes through the network in one forward pass, and every scan is trained against its
hardest positive (furthest same-subject scan) and hardest negative (closest other-subject
scan) in the batch. Distances are Euclidean on L2-normalised embeddings, and the loss is
averaged over the anchors that violate the margin.
```
python -m pointnet2.train.train_triplet -data_root /path/to/npy_faces \
    -model_checkpoint checkpoints/20191028_1000cls_model_best.pth.tar \
    -P 8 -K 4 -batches_per_epoch 500 -epochs 20 -lr 5e-5 --margin 0.3
```
Options:
* `-P` / `-K` (default 8 x 4 = 32 scans per batch). P=12, K=4 gives more negatives per
  anchor if GPU memory allows; `-num_points 2048` (default 4096) reduces memory.
* `--margin` (default 0.3). The log line `Hard n/B` counts anchors that still violate the
  margin; consider raising the margin to 0.4 if it falls below roughly 4/32.
* `-lr` (default 5e-5). Batch-hard gradients are sharper than random-triplet ones, so a
  high learning rate can collapse the embedding early.
* `-curvature_power` (default 0.1, the `curv**0.1` transform the pretrained model was
  trained with; use 1.0 if your curvature is already transformed).
* `-augment`: small random yaw rotation (applied to XYZ and normals) plus XYZ jitter.

Check a dataset with `python -m pointnet2.data.TripletNpyFaceDataset /path/to/npy_faces`.

## Results
**Comparison of verification rates at 0.1% FAR achieved on the FRGCv2 dataset**

Method  | N vs N | N vs Non-N | N vs All 
------------- | :-----------:|:-----------: |:-----------: 
Elaiwat(2015)  | 99.4 | 94.1 | 97.1 
Guo(2016)  | 99.9 | 97.2 | 99.0 
Lei(2016) | 99.9 | 96 | 98.3 
Gilani(2017) | 99.9 | 96.6 | 98.7 
Cai(2019) | 100 | 100 | 100 
Our work | 100 | 99.1 | 99.6 

**Comparison of rank-1 identification rates achieved on the overall FRGCv2 dataset**

| Method                | Rank-1 identification rate (%) |
| --------------------- | :----------------------------: |
| Elaiwat(2015)         |              97.1              |
| Li(2015)              |              96.3              |
| Lei(2016)             |              96.3              |
| Guo(2016)             |              97.0              |
| Gilani and Mian(2018) |             97.06              |
| Cai(2019)             |              100               |
| Our work              |             99.46              |

**Comparison of the rank-1 identification rate(%) on facial expression subset of Bosphorus dataset**

Method  | Rank-1 identification rate (%) 
------------- | :-----------:
Berretti(2013)  | 95.7 
Li(2015)  | 98.8 
Lei(2016) | 98.9 
Kim(2017) | 99.2 
Gilani and Mian(2018) | 96.18 
Cai(2019) | 99.75 
Bhople(2021) | 97.55 
Our work | 99.68 

**FRGCv2 ROC curve**

![](https://github.com/alfredtorres/3DFacePointCloudNet/blob/master/img/frgc_result.png)
