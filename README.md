# Ants_behavior — code accompanying the paper

This repository contains the code used to train the ant trajectory
PINN-PDE / Mechanism_NN model on experimental data, aggregate response
functions across reliable segments, and run the agent-based numerical
simulation built from the inferred response.

The final training is performed with **`num_t = 150`** time steps per
segment, which is the configuration used throughout the published
results.

---

## Requirements and Installation

- **Python**: 3.9 or later.
- **Libraries**:
  - `torch` (PyTorch with CUDA recommended for training; CPU is enough
    for visualization, averaging, and the numerical simulation)
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
- **External tools**:
  - **FFmpeg** for saving the MP4 animations produced by the
    visualization and the numerical simulation. The scripts look for
    `ffmpeg` in this order:
    1. The path in the `V2_FFMPEG_PATH` environment variable.
    2. `imageio_ffmpeg.get_ffmpeg_exe()` if the `imageio_ffmpeg` package
       is installed (`pip install imageio-ffmpeg`).
    3. `ffmpeg` on the system `PATH` (download from
       <https://ffmpeg.org/download.html>).
    If none of these resolve, the animations fall back to GIF.

---

## Data

The experimental trajectory data file `All_Data_frames.csv` is **not
distributed in this repository**. Download it from

> <https://zenodo.org/records/16797507>

and place it at the project root (next to this `README.md`):

```
Ants_behavior/
├── All_Data_frames.csv        <-- download here
├── README.md
├── ...
```

Each script reads the CSV path from the environment variable
`V2_DATA_PATH` if set, and falls back to the default location above.
Per-colony cached arrays are written into `cache/` on first use so
subsequent runs do not need to re-parse the CSV.

---

## Directory Structure

```
Ants_behavior/
├── README.md
├── All_Data_frames.csv                    # download from Zenodo (see Data)
├── v2_helpers.py                          # shared utilities (numdiff, losses, etc.)
├── v2_models.py                           # PINN_PDE and Mechanism_NN definitions
│
├── Model/
│   └── PDE/                               # trained model checkpoints (nt = 150)
│       ├── 620L_step-150s_model_pde_0
│       ├── 620L_step-150s_model_anten_0
│       └── ...                            # all colony × size × segment files
│
├── Training/
│   ├── main_training_and_visualization.py # train (if no model) + visualize
│   ├── first7_segments_averaging.py       # response averaging over first 7 reliable segments
│   └── Results/                           # plots and animations produced by the scripts
│
└── Simulation/
    ├── numerical_simulation_single_case.py
    └── Results/                           # response curve, trajectory MP4s, density, MSD
```

---

## How to Run

All three scripts are stand-alone — run them with `python <script>.py`
from anywhere; paths are resolved relative to the script location.

### 1. Train and / or visualize a segment

```bash
python Training/main_training_and_visualization.py
```

Behavior, per `(colony, size, segment index)` in `TARGETED_SEGMENTS`
(620L segments 0..7 by default):

1. If both `Model/PDE/{colony}{size}_step-150s_model_pde_{idx}` and the
   matching `_anten_{idx}` already exist, training is **skipped** and
   only the 7-artifact visualization runs.
2. Otherwise, training runs up to four attempts in the order
   `w_theta = 2.0, 2.0, 5.0, 5.0`:
   - The loop stops as soon as any attempt meets the final-save
     criteria.
   - If no attempt meets the criteria after all four, the best run so
     far (lowest `L_ang`) is saved as the fallback.

Final-save criteria:

| Quantity | Threshold |
|---|---|
| total optimizer iterations | `> 200` |
| `L_ang` (angular-velocity loss) | `< 8e-6` |
| `corr_tan` (NN·\|v\| ↔ measured `a_n` correlation) | `> 0.8` |
| `L_c` (concentration PDE loss) | `< 4e-6` |

Visualization artifacts (per segment) are saved in
`Training/Results/visualization_{colony}{size}_step-150s_idx{idx}/`:

1. `scatter_G_vs_dc.png` — two-panel response curve (NN line over the
   segment's reliable ∇⊥c support, masked outside) and predicted vs
   measured tangential-acceleration scatter
2. `trajectory.mp4` — agent trajectories with velocity quivers
3. `concentration_field.mp4` — pheromone-field overlay + agents
4. `mesh_and_trajectories.png` — PINN mesh points and trajectories
5. `trajectories_active.mp4` — agents coloured by the active/inactive mask

### 2. Average the response over the first 7 reliable segments

```bash
python Training/first7_segments_averaging.py
```

Walks segments chronologically for each `(colony, size)` in
`COLONIES` (620L by default) and keeps the first 7 segments whose
reliable ∇⊥c support spans `[-1.5, +1.5]`. The kept segments are
averaged into a mean ± std response curve and saved to
`Training/Results/`:

- `{tag}_anten.{jpg,eps}` — mean response curve with orange PR-outlier
  overlay
- `{tag}_PR-curve.{jpg,eps}` — predicted vs measured tangential
  acceleration
- `{tag}_ver1.9.jpg` — two-panel response (wide + zoom)
- `{tag}_prob.{jpg,eps}` — spatial density distribution
- `summary.csv` — per-case curve statistics

### 3. Run the numerical simulation (single case)

```bash
python Simulation/numerical_simulation_single_case.py
```

Runs one agent-based trial with the colony-mean parameters

```
I_ave = 1.0,  I_att = 0.1,  lambda_att = 0.6,  noise_std = 0.1
```

over `T_end = 200 s` with 65 agents on a periodic `[-0.5, 0.5]^2`
domain. Outputs to `Simulation/Results/` (still plots are saved as
both `.jpg` and `.eps`):

- `Simulation_response_function_*` — response curve `f(∇⊥c)`
- `Simulation_Aggregation_Probability_*` — agent density distribution
  with `P(ρ > 1)` annotated
- `Simulation_MSD_*` — log-log mean MSD with slope-1 / slope-2 guides
  and the fitted diffusion exponent
- `Simulation_trajectory_*.mp4` — periodic-wrapped trajectory with
  pheromone-field overlay
- `Simulation_continuous_trajectory_*.mp4` — unwrapped trajectory
  animation

---

## Reproducing the paper case

The default settings in all three scripts target **620L** (colony 620,
large ants) which is the representative case used throughout the
paper. All eight trained model checkpoints for 620L are provided in
`Model/PDE/`, so running step 1 above produces the visualization
without any retraining, step 2 produces the averaged response curve
and density plot, and step 3 produces the numerical simulation
artifacts at the colony-mean parameters.
