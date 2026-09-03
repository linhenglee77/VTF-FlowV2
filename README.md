#  VTF-Flow

PyTorch implementation of **VTF-Flow**, a vehicle-conditioned
terrain-feasibility-guided conditional Flow Matching planner for unstructured
off-road environments. VTF-Flow constructs a continuous terrain--vehicle
kinematic (TVK) potential from RELLIS-3D LiDAR geometry and semantic labels and
uses the potential both as a bounded training regularizer and as an in-flow
guidance signal during trajectory generation.

This repository accompanies the manuscript on trajectory planning in
unstructured off-road environments. Conditional Flow Matching is used as the
multi-candidate generation backbone; the repository's main focus is the
continuous vehicle-conditioned feasibility representation and its integration
into the Flow evolution.

## Scope and interpretation

- A trajectory is represented in planning-ego coordinates as
  `[H, 3] = [(x_i, y_i, z_i)]`, with `x` forward, `y` left, and `z` up.
- The paper protocol uses `H=10`, `dt=0.5 s`, and a `5 s` prediction horizon.
- Flow-based methods generate `K=8` candidates with a 16-step Euler solver.
- The recorded RELLIS-3D future is a reference driving trajectory recovered
  from poses. It is not assumed to be a globally optimal or guaranteed-safe
  path.
- TVK potentials are relative planning diagnostics, not calibrated safety
  probabilities or binary safety certificates.
- Curvature and lateral acceleration are kinematic approximations. RELLIS-3D
  does not provide the vehicle and tire--terrain parameters needed for a full
  dynamics or stability model.

## Repository layout

```text
.
├── configs/          # Frozen experiment, terrain-field, and split settings
├── datasets/         # RELLIS-3D parsing, cache construction, and SceneBatch adapter
├── models/           # Scene encoder and Flow/regression networks
├── planners/         # Classical, regression, Flow, and VTF-Flow planners
├── terrain/          # Static, vehicle-conditioned, and TVK potentials
├── guidance/         # In-flow feasibility guidance and coordination operators
├── metrics/          # Trajectory and feasibility metrics
├── evaluation/       # Common evaluators and sequence-level aggregation
├── visualization/    # Plotting utilities
├── scripts/          # Preprocessing, training, evaluation, and figure entry points
├── tests/            # Unit and synthetic numerical tests
├── docs/             # Design, data-audit, coordinate, and experiment documentation
└── reproducibility/  # Reference values for checking a reproduced run
```

## What is required to reproduce the paper

### Files tracked in this repository

Keep the following directories and files in the public source release:

- `configs/`, including `unified_h10_benchmark.json`,
  `sequence_holdout_robustness.json`,
  `sequence_holdout_full_benchmark.json`,
  `rellis3d_terrain_field.json`,
  `rellis3d_os1_to_planning_ego.json`, and the frozen validation-demonstration
  envelope;
- `datasets/`, `models/`, `planners/`, `terrain/`, `guidance/`, `metrics/`,
  `evaluation/`, and `closed_loop/`;
- `scripts/`, `tests/`, and the methodological documentation in `docs/`;
- `pyproject.toml`, `requirements.txt`, `.gitignore`, and this README;
- `reproducibility/reference_main_table.csv` and
  `reproducibility/reference_field_validation.csv`, with hashes recorded in
  `reproducibility/SHA256SUMS`.

### External data required for a from-scratch run

Download RELLIS-3D from the
[official project repository](https://github.com/unmannedlab/RELLIS-3D) and
follow its terms of use. The VTF-Flow repository does **not** redistribute the
raw RGB images, point clouds, semantic labels, poses, or calibration files.
The upstream project states that its dataset and code are provided under
CC BY-NC-SA 3.0; users remain responsible for checking the current upstream
terms.

The preprocessing script expects the following layout:

```text
<DATA_ROOT>/
└── processed/
    ├── metadata/
    │   ├── pt_train.lst
    │   ├── pt_val.lst
    │   └── pt_test.lst
    └── Rellis-3D/
        ├── 00000/
        ├── 00001/
        ├── 00002/
        ├── 00003/
        └── 00004/
            ├── os1_cloud_node_kitti_bin/*.bin
            ├── os1_cloud_node_semantickitti_label_id/*.label
            ├── pylon_camera_node/*.jpg
            ├── poses.txt
            ├── calib.txt
            └── camera_info.txt
```

Each sequence must contain the same modality directories. RGB images are only
needed for camera-view qualitative figures; the core training and benchmark
require Ouster point clouds, semantic labels, poses, and split metadata.

### Optional large artifact bundle

A faster evaluation-only reproduction can use a separately archived artifact
bundle containing:

```text
trajectory_cache_h150_s5/
├── dataset_config.json
├── train/{bev.npy,trajectory.npy,goal.npy,manifest.csv}
├── val/{bev.npy,trajectory.npy,goal.npy,manifest.csv}
└── test/{bev.npy,trajectory.npy,goal.npy,manifest.csv}

outputs/sequence_holdout_robustness/
└── runs/holdout_0000{0,1,2}/seed_{0,1,2}/{FLOW,VTF_V2}/...

outputs/sequence_holdout_full_benchmark/checkpoints/
└── regression checkpoints for all three folds and seeds
```

The local reference cache is approximately 140 MB and the frozen robustness
output tree is approximately 148 MB. These generated binary artifacts are
ignored by Git and should be distributed through a versioned GitHub Release,
Git LFS, or an archival repository rather than committed to ordinary Git
history. Preserve the directory hierarchy and publish SHA-256 checksums with
the archive. A public artifact URL has not been hard-coded here because no
stable release identifier has yet been assigned.

## Environment

The reference run used:

| Component | Reference environment |
|---|---|
| OS | Windows 11 |
| Python | 3.9.23 |
| PyTorch | 2.8.0 + CUDA 12.9 |
| GPU | NVIDIA GeForce RTX 5060, 8 GB |
| NumPy | 1.26.3 |
| pandas | 2.3.2 |
| SciPy | 1.13.1 |
| Matplotlib | 3.9.4 |
| Pillow | 11.0.0 |
| PyYAML | 6.0.3 |

Other recent PyTorch 2.x environments should work, but small numerical
differences may occur across hardware, CUDA, and library versions. The paper
reports sequence-level means and standard deviations rather than requiring
bitwise-identical floating-point output.

Create an environment and install the repository from its root:

```bash
conda create -n vtf-flow python=3.9 -y
conda activate vtf-flow
python -m pip install --upgrade pip
python -m pip install -e .
```

For a CPU-only installation, install an appropriate PyTorch build first and
then run `python -m pip install -e . --no-deps` followed by the remaining
packages in `requirements.txt`.

## Reproduction workflow

All commands below are run from the repository root. Paths are supplied on the
command line; the code contains no required machine-specific dataset path.

### 1. Run unit tests

```bash
python tests/run_tests.py
```

The suite covers interfaces, pose/trajectory construction, terrain-field
queries, vehicle conditioning, Flow Matching, guidance, kinematic terms,
metrics, and cache parsing.

### 2. Build the leakage-controlled trajectory cache

```bash
python -m TerraFlow.scripts.build_rellis3d_cache \
  --data-root <DATA_ROOT>/processed \
  --output-dir <CACHE_ROOT>/trajectory_cache_h150_s5 \
  --horizon 150 \
  --trajectory-stride 5 \
  --isolation 150 \
  --grid-size 64
```

The expected cache manifest is:

```text
horizon_frames: 150
trajectory_stride: 5
trajectory_points_excluding_origin: 30
BEV shape: [3, 64, 64]
BEV channels: traversable_fraction, obstacle_density, mean_height
counts: train=6750, val=1813, test=2443
normalization scales: [24 m, 12 m, 3 m]
```

The 30-point cached future is sampled at nominal 0.5 s intervals. The paper
planner evaluates the first `H=10` points, corresponding to 5 s.

### 3. Reproduce the independent feasibility-potential diagnostic

This diagnostic uses all 1,909 eligible trajectories from sequence `00004`.
It is independent of the three outer held-out benchmark sequences.

```bash
python -m TerraFlow.scripts.validate_field_guidance \
  --data-root <DATA_ROOT> \
  --cache-root <CACHE_ROOT>/trajectory_cache_h150_s5 \
  --split train \
  --sequence 00004 \
  --sensor-to-ego configs/rellis3d_os1_to_planning_ego.json \
  --gradient-scenes 128 \
  --descent-scenes 128 \
  --output outputs/experiments/field_guidance_validation.csv
```

Compare the principal values with
`reproducibility/reference_field_validation.csv`. The complete output also
contains component contributions, gradient distributions, perturbation
stability, and 5/10/20-step field-only descent diagnostics.

### 4. Train Flow Matching and VTF-Flow on all outer folds

This is the main computational step. **Pass all seeds explicitly**; the script
defaults to the screening seed only when `--seeds` is omitted.

```bash
python -m TerraFlow.scripts.run_sequence_holdout_robustness \
  --protocol configs/sequence_holdout_robustness.json \
  --benchmark configs/unified_h10_benchmark.json \
  --cache-root <CACHE_ROOT>/trajectory_cache_h150_s5 \
  --data-root <DATA_ROOT> \
  --output-root outputs/sequence_holdout_robustness \
  --folds 00000 00001 00002 \
  --seeds 0 1 2
```

The frozen protocol is:

- development validation sequence: `00003`;
- outer held-out test sequences: `00000`, `00001`, and `00002`;
- training seeds: `0`, `1`, and `2`;
- 40 training epochs per fold and seed;
- VTF-Flow guidance: `eta=0.075`, late-strong schedule, `gamma=1`,
  three-point trajectory smoothing, and terminal endpoint projection;
- identical initial Gaussian candidates for paired Flow/VTF-Flow evaluation.

The expected test counts are 1,797, 1,719, and 3,547 scenes, respectively
(7,063 total). The independent statistical unit is the held-out sequence;
seeds are technical training replicates averaged within each sequence.

To re-evaluate existing frozen Flow/VTF-Flow checkpoints and predictions, add
`--skip-training` and retain the generated output hierarchy.

### 5. Run the strict five-method benchmark

This stage evaluates Constant Velocity, A* terrain planning, deterministic
regression, unguided Flow Matching, and VTF-Flow with the same scene indices,
goal definition, terrain/TVK evaluator, and aggregation protocol. It reuses the
Flow and VTF-Flow archives from Step 4 and trains the regression baseline.

```bash
python -m TerraFlow.scripts.run_sequence_holdout_full_benchmark \
  --config configs/sequence_holdout_full_benchmark.json \
  --cache-root <CACHE_ROOT>/trajectory_cache_h150_s5 \
  --data-root <DATA_ROOT> \
  --output-root outputs/sequence_holdout_full_benchmark
```

If the artifact bundle already supplies every regression checkpoint, append
`--skip-regression-training`.

Principal outputs:

```text
outputs/sequence_holdout_full_benchmark/
├── effective_protocol.json
├── main_table.csv
├── main_table_numeric.csv
├── main_table.tex
├── per_sequence_summary.csv
├── run_summary.csv
├── vtf_flow_pairwise_sequence_effects.csv
├── astar_coverage_report.json
└── benchmark_report_zh.md
```

Compare `main_table_numeric.csv` with
`reproducibility/reference_main_table.csv`. The reported values are unweighted
macro means and sample standard deviations across the three held-out
sequences.

### 6. Generate the main benchmark and real-data figures

```bash
python -m TerraFlow.scripts.plot_sequence_holdout_full_benchmark

python -m TerraFlow.scripts.render_holdout_manuscript_qualitative \
  --data-root <DATA_ROOT> \
  --cache-root <CACHE_ROOT>/trajectory_cache_h150_s5 \
  --output-root outputs/sequence_holdout_full_benchmark/figures/heldout_real_data \
  --raw-field-config configs/rellis3d_terrain_field.json \
  --sensor-transform configs/rellis3d_os1_to_planning_ego.json \
  --rebuild-fields
```

The qualitative script uses frozen seed-0 predictions and always plots
candidate 0 for both generative planners. It does not use GT to select the
displayed candidate. One scene per held-out sequence is selected using recorded
path length and camera-projection visibility; the terrain-decomposition scene
is selected using static-potential heterogeneity rather than method gain.

### 7. Optional component ablation used in the manuscript

The sequence-`00004` component ablation is separate from the outer-holdout main
benchmark. Reproduce its base checkpoints and then the TVK variants with:

```bash
python -m TerraFlow.scripts.run_final_experiments \
  --cache-root <CACHE_ROOT>/trajectory_cache_h150_s5 \
  --config configs/final_experiments.json \
  --output-root outputs/final_experiments

python -m TerraFlow.scripts.run_final_tvk_validation \
  --cache-root <CACHE_ROOT>/trajectory_cache_h150_s5 \
  --config configs/final_tvk_validation.json \
  --baseline-root outputs/final_experiments \
  --output-root outputs/final_experiments_tvk_final
```

The separately optimized `VTF-Flow w/o training regularisation` variant is an
accuracy--feasibility diagnostic, not a strict one-factor ablation.

## Expected main result

The complete reference table is stored in
`reproducibility/reference_main_table.csv`. At four decimal places, the main
rows are:

| Method | ADE-0 (m) | minADE@K (m) | TVK potential | Terrain violation | q80 candidate rate | GCCR@K |
|---|---:|---:|---:|---:|---:|---:|
| Constant Velocity | 0.5783 | 0.5783 | 0.7060 | 0.5832 | 0.0106 | 0.0106 |
| A* terrain planner | 1.6191 | 1.6191 | 0.9283 | 0.5505 | 0.0104 | 0.0104 |
| Deterministic regression | 0.1618 | 0.1618 | 0.6833 | 0.5704 | 0.3744 | 0.3744 |
| Flow Matching | 0.1684 | 0.1402 | 0.6856 | 0.5707 | 0.3753 | 0.4040 |
| **VTF-Flow** | 0.1662 | **0.1387** | **0.6788** | 0.5673 | **0.3845** | **0.4105** |

These results support a balanced improvement in candidate coverage and the
defined terrain--kinematic quality relative to the matched unguided Flow
baseline. They do not establish universal superiority on every metric or
physical-vehicle safety.

## Reproducibility notes

- Never report a run without archiving its `effective_protocol.json`.
- Do not change candidate count, initial-noise construction, goal definition,
  terrain evaluator, or sequence aggregation when making a paired comparison.
- Do not fit q80/q90/q95 compliance envelopes on the outer test sequences. The
  frozen values in `configs/validation_demonstration_envelope.json` were derived
  from development sequence `00003`.
- A* out-of-map and disconnected cases are retained and explicitly recorded in
  `astar_coverage_report.json`; they are not silently removed.
- Small GPU-dependent differences are expected. Material deviations from the
  reference table should first be checked against the cache manifest, executed
  seeds, fold assignment, PyTorch/CUDA version, and effective protocol.

## Dataset citation

Please cite the original RELLIS-3D dataset in addition to the VTF-Flow paper:

```bibtex
@inproceedings{jiang2021rellis3d,
  title     = {RELLIS-3D Dataset: Data, Benchmarks and Analysis},
  author    = {Jiang, Peng and Osteen, Philip and Wigness, Maggie and Saripalli, Srikanth},
  booktitle = {2021 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2021}
}
```

## License and archival release

No software license has yet been selected for this local pre-submission copy.
The repository owner must add a `LICENSE` file before public release; without
one, normal copyright restrictions apply and reuse is not clearly granted.
The RELLIS-3D licence applies separately to the upstream data.

For the accepted-paper release, create a versioned source release and archive
that release in a DOI-minting repository. Add the software DOI, artifact URL,
version tag, and SHA-256 manifest here and in the manuscript's Code and Data
Availability statement.

## Contact

For reproducibility questions, please open a GitHub issue after the public
repository URL is assigned.
