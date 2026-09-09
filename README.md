# Ants_collective_behavior

## Code for "Ants aggregate with avoidance of weak spatial gradients of trajectory-derived cues"

This repository contains the code used to train the ant-trajectory
PINN-PDE / Mechanism_NN framework on experimental data, aggregate
perception-response functions across reliable trajectory segments, and
run the agent-based numerical simulation based on the inferred turning
response.

In the terminology of the manuscript, `PINN_PDE` represents the
trajectory-derived cue field, whereas `Mechanism_NN` represents the
perception-response function relating the normal cue concentration
gradient to ant turning.

The final training uses **`num_t = 150`** sampled time points per
segment, corresponding to approximately 150 s at the 1 s analysis
interval used for neural-network training.

Some internal variable names are retained from earlier versions of the
code for compatibility. In particular, `I_ave` corresponds to the
avoidance intensity \(I_{\mathrm{avo}}\) in the manuscript,
`lambda_att` corresponds to the response threshold \(\lambda\), and
variables such as `accel_tan` and `corr_tan` refer to the normal
acceleration component \(a_n\), defined perpendicular to the
instantaneous velocity.

---

## Requirements and Installation

- **Python**: 3.9 or later.
- **Libraries**:
  - `torch` (PyTorch with CUDA recommended for training; CPU is
    sufficient for visualization, averaging, and the numerical
    simulation)
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
- **External tools**:
  - **FFmpeg** for saving the MP4 animations produced by the
    visualization and numerical-simulation scripts. The scripts look for
    `ffmpeg` in this order:
    1. The path in the `V2_FFMPEG_PATH` environment variable.
    2. `imageio_ffmpeg.get_ffmpeg_exe()` if the `imageio_ffmpeg`
       package is installed (`pip install imageio-ffmpeg`).
    3. `ffmpeg` on the system `PATH` (download from
       <https://ffmpeg.org/download.html>).

    If none of these resolve, the animations fall back to GIF.

---

## Data

The experimental trajectory data file `All_Data_frames.csv` is **not
distributed in this repository**. Download it from

> <https://zenodo.org/records/16797507>

and place it at the project root next to this `README.md`:

```text
Ants_collective_behavior/
├── All_Data_frames.csv        <-- download here
├── README.md
├── ...
```

Each script reads the CSV path from the environment variable
`V2_DATA_PATH` if set and otherwise uses the default location above.

Per-colony cached arrays are written into `cache/` on first use so that
subsequent runs do not need to re-parse the CSV.

---

## Directory Structure

```text
Ants_collective_behavior/
├── README.md
├── All_Data_frames.csv                    # download from Zenodo (see Data)
├── v2_helpers.py                          # shared utilities, numerical differentiation, losses
├── v2_models.py                           # PINN_PDE and Mechanism_NN definitions
│
├── Model/
│   └── PDE/                               # trained model checkpoints (num_t = 150)
│       ├── 620L_step-150s_model_pde_0
│       ├── 620L_step-150s_model_anten_0
│       └── ...                            # colony × size × segment checkpoint files
│
├── Training/
│   ├── main_training_and_visualization.py # train if needed, then visualize
│   ├── first7_segments_averaging.py       # average first 7 reliable segments
│   └── Results/                           # plots and animations produced by the scripts
│
└── Simulation/
    ├── numerical_simulation_single_case.py
    └── Results/                           # response curve, trajectories, density, MSD
```

---

## How to Run

All three scripts are stand-alone. Run them with `python <script>.py`
from anywhere; paths are resolved relative to the script location.

### 1. Train and/or visualize a segment

```bash
python Training/main_training_and_visualization.py
```

For each `(colony, size, segment index)` specified in
`TARGETED_SEGMENTS` (620L segments 0–7 by default):

1. If both
   `Model/PDE/{colony}{size}_step-150s_model_pde_{idx}` and the
   corresponding `_anten_{idx}` checkpoint already exist, training is
   **skipped** and the 5-artifact visualization is generated.
2. Otherwise, training runs up to four attempts in the order

   ```text
   w_theta = 2.0, 2.0, 5.0, 5.0
   ```

   - The loop stops as soon as an attempt satisfies the final-save
     criteria.
   - If none of the four attempts satisfies the criteria, the run with
     the lowest `L_ang` is saved as the fallback.

### Final-save criteria

| Quantity | Threshold |
|---|---:|
| total optimizer iterations | `> 200` |
| `L_ang` (normal-acceleration loss) | `< 8e-6` |
| `corr_tan` (predicted vs. measured \(a_n\) correlation; legacy variable name) | `> 0.8` |
| `L_c` (cue-field PDE loss) | `< 4e-6` |

Visualization artifacts for each segment are saved in

```text
Training/Results/visualization_{colony}{size}_step-150s_idx{idx}/
```

and consist of:

1. `scatter_G_vs_dc.png`  
   Two-panel plot containing the inferred perception-response curve over
   the segment's reliable \(\nabla_\perp c\) support and the
   predicted-vs.-measured normal-acceleration scatter.

2. `trajectory.mp4`  
   Ant trajectories with velocity vectors.

3. `concentration_field.mp4`  
   Reconstructed trajectory-derived cue field with ant positions and
   velocity vectors.

4. `mesh_and_trajectories.png`  
   Spatial points used for the PINN cue-field reconstruction together
   with the ant trajectories.

5. `trajectories_active.mp4`  
   Ant trajectories colored according to whether the observations are
   retained for perception-response training, based on the
   active/inactive and wall-following criteria.

If FFmpeg is unavailable, the animation files are saved as GIF instead
of MP4.

---

### 2. Average the response over the first 7 reliable segments

```bash
python Training/first7_segments_averaging.py
```

The script processes segments chronologically for each `(colony, size)`
specified in `COLONIES` (620L by default) and retains the first 7
segments whose reliable \(\nabla_\perp c\) support spans

```text
[-1.5, +1.5]
```

The retained segment-level perception-response functions are averaged
to obtain the mean ± standard deviation response for each experimental
case.

Outputs are saved to `Training/Results/`:

- `{tag}_anten.{jpg,eps}`  
  Mean perception-response curve with standard-deviation band and
  PR-outlier overlay.

- `{tag}_PR-curve.{jpg,eps}`  
  Predicted vs. measured normal acceleration.

- `{tag}_ver1.9.jpg`  
  Two-panel perception-response plot showing wide and zoomed ranges.

- `{tag}_prob.{jpg,eps}`  
  Local ant-density distribution used to calculate the local spatial
  aggregation probability \(P(\rho > 1)\).

- `summary.csv`  
  Per-case statistics extracted from the averaged response functions,
  trajectory data, and local-density analysis.

---

### 3. Run the numerical simulation

```bash
python Simulation/numerical_simulation_single_case.py
```

The script runs a single agent-based simulation using the reference
parameter values

```text
I_ave = 1.0
I_att = 0.1
lambda_att = 0.6
noise_std = 0.1
```

where:

- `I_ave` is the legacy code identifier corresponding to the avoidance
  intensity \(I_{\mathrm{avo}}\);
- `I_att` is the attraction intensity \(I_{\mathrm{att}}\);
- `lambda_att` is the response threshold \(\lambda\); and
- `noise_std` is the standard deviation of the stochastic turning term.

The simulation uses:

```text
number of agents = 65
T_end            = 200 s
dt_sim           = 0.1 s
domain           = [-0.5, 0.5] × [-0.5, 0.5]
boundary         = periodic
```

The simulation implements the non-monotonic response function
\(f(\nabla_\perp c)\) together with stochastic angular noise and a
trajectory-derived cue field generated by the moving agents.

Outputs are written to `Simulation/Results/`. Still plots are saved as
both `.jpg` and `.eps`.

- `Simulation_response_function_*`  
  Non-monotonic response function \(f(\nabla_\perp c)\).

- `Simulation_Aggregation_Probability_*`  
  Agent-density distribution with the local spatial aggregation
  probability \(P(\rho > 1)\).

- `Simulation_MSD_*`  
  Log-log mean squared displacement with slope-1 and slope-2 reference
  lines and the fitted diffusion exponent.

- `Simulation_trajectory_*.mp4`  
  Periodically wrapped agent trajectories with the modeled
  trajectory-derived cue field.

- `Simulation_continuous_trajectory_*.mp4`  
  Corresponding unwrapped trajectory animation.

If FFmpeg is unavailable, animation files are saved as GIF.

---

## Reproducing the Representative Manuscript Case

The default settings in the training, averaging, and simulation scripts
target **620L** (colony 620, major worker class), which is the
representative experimental case used to illustrate the analysis
pipeline in the manuscript.

Eight trained trajectory-segment checkpoints for 620L are provided in
`Model/PDE/`.

Therefore:

1. Running

   ```bash
   python Training/main_training_and_visualization.py
   ```

   generates the segment-level visualizations without retraining when
   the corresponding checkpoints are present.

2. Running

   ```bash
   python Training/first7_segments_averaging.py
   ```

   selects the first seven reliable segments and generates the averaged
   perception-response curve and local-density analysis.

3. Running

   ```bash
   python Simulation/numerical_simulation_single_case.py
   ```

   generates the single-case agent-based simulation using the reference
   response parameters.

---

## Terminology and Legacy Variable Names

The code was developed over several stages of the analysis, and some
internal variable names predate the terminology used in the final
manuscript. These identifiers have been retained to preserve
compatibility with the trained models and analysis scripts.

The principal correspondences are:

| Code identifier or legacy term | Manuscript terminology |
|---|---|
| `I_ave` | avoidance intensity \(I_{\mathrm{avo}}\) |
| `I_att` | attraction intensity \(I_{\mathrm{att}}\) |
| `lambda_att` | response threshold \(\lambda\) |
| `accel_tan` | normal acceleration \(a_n\) |
| `corr_tan` | correlation between predicted and measured normal acceleration |
| concentration / pheromone field in legacy code comments | trajectory-derived cue field |

These legacy names do not imply a different quantity from that described
in the manuscript; they are retained only as internal code identifiers.

---

## Data and Code Availability

The experimental trajectory dataset used by these scripts is available
from Zenodo:

> <https://zenodo.org/records/16797507>

The present repository contains the training, response-averaging, and
agent-based simulation code used for the analyses described above.
