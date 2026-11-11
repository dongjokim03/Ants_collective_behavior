# Ants_collective_behavior

## Code for "Local aggregation and avoidance of weak spatial gradients in collective dynamics"

This repository contains the code used to train the ant-trajectory
PINN-PDE / Mechanism_NN model on experimental trajectory data, average
the inferred perception-response functions across reliable segments,
and run the agent-based numerical simulation based on the inferred
response.

In the terminology of the manuscript, `PINN_PDE` reconstructs the
trajectory-derived cue field, and `Mechanism_NN` infers the
perception-response function relating the normal cue concentration
gradient to directional turning.

The final training uses **`num_t = 150`** sampled time points per
segment, corresponding to an approximately 150-s trajectory segment at
the 1-s analysis interval used for neural-network training.

Some internal variable names are retained from earlier versions of the
code. Their correspondence to the terminology used in the manuscript
is summarized below.

| Code identifier | Manuscript terminology |
|---|---|
| `I_ave` | avoidance intensity $I_{\mathrm{avo}}$ |
| `I_att` | attraction intensity $I_{\mathrm{att}}$ |
| `lambda_att` | response threshold $\lambda$ |
| `accel_tan` | normal acceleration $a_n$ |
| `corr_tan` | correlation between predicted and measured normal acceleration |
| concentration / pheromone field in legacy code comments | trajectory-derived cue field |

These names are retained only as internal code identifiers and do not
represent quantities different from those defined in the manuscript.

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
  - **FFmpeg** for saving MP4 animations. The scripts look for `ffmpeg`
    in the following order:
    1. The path specified by the `V2_FFMPEG_PATH` environment variable.
    2. `imageio_ffmpeg.get_ffmpeg_exe()` if the optional
       `imageio_ffmpeg` package is installed
       (`pip install imageio-ffmpeg`).
    3. `ffmpeg` available on the system `PATH`
       (<https://ffmpeg.org/download.html>).

    If FFmpeg cannot be resolved, animations are saved as GIF files.

---

## Data

The experimental trajectory data file `All_Data_frames.csv` is **not
distributed in this repository**. Download it from

> <https://zenodo.org/records/16797507>

and place it in the project root next to this `README.md`:

```text
Ants_collective_behavior/
├── All_Data_frames.csv        <-- download here
├── README.md
├── ...
```

Each script reads the CSV path from the environment variable
`V2_DATA_PATH` if it is set and otherwise uses the default location
above.

Cached arrays used for repeated analyses are stored in `cache/` so that
the raw CSV does not need to be parsed again for every run.

---

## Directory Structure

```text
Ants_collective_behavior/
├── README.md
├── All_Data_frames.csv                    # download from Zenodo (see Data)
├── v2_helpers.py                          # shared utilities, differentiation, and losses
├── v2_models.py                           # PINN_PDE and Mechanism_NN definitions
├── cache/                                 # cached trajectory and preprocessing data
│
├── Model/
│   └── PDE/                               # trained PINN_PDE and Mechanism_NN checkpoints
│       ├── 620L_step-150s_model_pde_0
│       ├── 620L_step-150s_model_anten_0
│       └── ...                            # colony × size × segment checkpoint files
│
├── Training/
│   ├── main_training_and_visualization.py # train if needed and visualize each segment
│   ├── first7_segments_averaging.py       # average the first 7 reliable segments
│   └── Results/                           # training and averaging outputs
│
└── Simulation/
    ├── numerical_simulation_single_case.py
    └── Results/                           # simulation response, density, MSD, and animations
```

---

## How to Run

The three analysis scripts can be run with `python <script>.py` from
any location because their paths are resolved relative to the script
location.

### 1. Train and/or visualize a trajectory segment

```bash
python Training/main_training_and_visualization.py
```

The default `TARGETED_SEGMENTS` setting processes segments 0–7 of the
representative **620L** case.

For each `(colony, size, segment index)`:

1. If both

   ```text
   Model/PDE/{colony}{size}_step-150s_model_pde_{idx}
   ```

   and the corresponding

   ```text
   Model/PDE/{colony}{size}_step-150s_model_anten_{idx}
   ```

   already exist, training is skipped and the saved models are used
   directly for visualization.

2. Otherwise, training is attempted up to four times using

   ```text
   w_theta = 2.0, 2.0, 5.0, 5.0
   ```

   in this order.

   - Training stops when an attempt satisfies the final-save criteria.
   - If none of the four attempts satisfies all criteria, the run with
     the lowest `L_ang` is saved as the fallback.

### Final-save criteria

| Quantity | Threshold |
|---|---:|
| total optimizer iterations | `> 200` |
| `L_ang` (normal-acceleration loss) | `< 8e-6` |
| `corr_tan` (predicted–measured $a_n$ correlation) | `> 0.8` |
| `L_c` (cue-field PDE loss) | `< 4e-6` |

For each segment, five visualization artifacts are saved in

```text
Training/Results/visualization_{colony}{size}_step-150s_idx{idx}/
```

1. `scatter_G_vs_dc.png`  
   Two-panel figure showing the inferred perception-response function
   over the reliable $\nabla_\perp c$ range and the
   predicted-versus-measured normal-acceleration scatter.

2. `trajectory.mp4`  
   Ant trajectories with instantaneous velocity vectors.

3. `concentration_field.mp4`  
   Reconstructed trajectory-derived cue field together with ant
   positions and velocity vectors.

4. `mesh_and_trajectories.png`  
   Spatial points used for cue-field reconstruction together with the
   ant trajectories.

5. `trajectories_active.mp4`  
   Trajectory visualization of the observations used for
   perception-response inference. Observations satisfying the activity
   criterion and not classified as wall-following are shown separately
   from excluded observations.

If FFmpeg is unavailable, the animation outputs are saved as GIF files.

---

### 2. Average the response over the first 7 reliable segments

```bash
python Training/first7_segments_averaging.py
```

The script processes trajectory segments chronologically for each
`(colony, size)` specified in `COLONIES`. The default setting processes
the representative 620L case.

For each segment, the reliable range of $\nabla_\perp c$ is defined
from the 0.5th–99.5th percentiles of the sampled gradient values. The
script retains the first seven segments whose reliable support covers

```text
[-1.5, +1.5]
```

and averages their inferred perception-response functions.

The mean and standard deviation are calculated across the retained
segment-level response functions.

Outputs are saved in `Training/Results/`:

- `{tag}_anten.{jpg,eps}`  
  Mean perception-response function with its standard-deviation band
  and the identified prediction-response outliers.

- `{tag}_PR-curve.{jpg,eps}`  
  Predicted-versus-measured normal-acceleration scatter.

- `{tag}_ver1.9.jpg`  
  Two-panel perception-response figure showing the wide and zoomed
  gradient ranges.

- `{tag}_prob.{jpg,eps}`  
  Distribution of local ant density used to calculate the local spatial
  aggregation probability $P(\rho > 1)$.

- `summary.csv`  
  Per-case response parameters and trajectory-derived summary
  statistics.

---

### 3. Run the agent-based numerical simulation

```bash
python Simulation/numerical_simulation_single_case.py
```

This script runs a single constant-speed agent-based simulation using
the reference parameter values

```text
I_ave      = 1.0
I_att      = 0.1
lambda_att = 0.6
noise_std  = 0.1
```

where

- `I_ave` corresponds to the avoidance intensity
  $I_{\mathrm{avo}}$;
- `I_att` corresponds to the attraction intensity
  $I_{\mathrm{att}}$;
- `lambda_att` corresponds to the response threshold $\lambda$; and
- `noise_std` is the standard deviation of the stochastic turning term.

The default simulation uses

```text
number of agents = 65
agent speed      = 0.02 m/s
T_end            = 200 s
dt_sim           = 0.1 s
domain           = [-0.5, 0.5] × [-0.5, 0.5]
boundary         = periodic
```

Each agent deposits the modeled cue along its trajectory and changes
its heading according to the non-monotonic response
$f(\nabla_\perp c)$ together with stochastic angular noise.

Outputs are saved in `Simulation/Results/`. Static plots are saved in
both `.jpg` and `.eps` formats.

- `Simulation_response_function_*`  
  Non-monotonic perception-response function
  $f(\nabla_\perp c)$.

- `Simulation_Aggregation_Probability_*`  
  Local-density distribution with the local spatial aggregation
  probability $P(\rho > 1)$.

- `Simulation_MSD_*`  
  Log-log mean squared displacement with slope-1 and slope-2 reference
  lines and the fitted diffusion exponent.

- `Simulation_trajectory_*.mp4`  
  Periodically wrapped agent trajectories together with the modeled
  trajectory-derived cue field.

- `Simulation_continuous_trajectory_*.mp4`  
  Corresponding unwrapped trajectories used to visualize continuous
  motion across the periodic boundaries.

If FFmpeg is unavailable, animation outputs are saved as GIF files.

---

## Reproducing the Representative Manuscript Case

The default settings of the training and averaging scripts target
**620L** (colony 620, major worker class), which is the representative
experimental case used to illustrate the inference procedure in the
manuscript.

Trained `PINN_PDE` and `Mechanism_NN` checkpoint files for 620L
segments 0–7 are provided in `Model/PDE/`.

Therefore:

1. Running

   ```bash
   python Training/main_training_and_visualization.py
   ```

   generates the segment-level visualizations directly from the
   provided checkpoints unless the target checkpoint files are absent.

2. Running

   ```bash
   python Training/first7_segments_averaging.py
   ```

   selects the first seven reliable trajectory segments and generates
   the averaged perception-response function and local-density
   analysis.

3. Running

   ```bash
   python Simulation/numerical_simulation_single_case.py
   ```

   generates the single-case agent-based simulation using the reference
   response parameters specified above.

The default scripts are provided primarily to reproduce the
representative analysis workflow. Other colony and worker-size cases
can be processed by changing `TARGETED_SEGMENTS` or `COLONIES` to the
corresponding checkpoint and trajectory-data entries.

---

## Data and Code Availability

The experimental trajectory data used by the scripts are available at

> <https://zenodo.org/records/16797507>

This repository provides the neural-network training, segment
averaging, visualization, and agent-based simulation code used for the
analyses described in the manuscript.
