# <Thesis title>

Code, data, and figures for my thesis, *<full title>* (<degree>, <university>, <year>).

<!-- 2–4 sentences: the problem (action decisions from multimodal V2I sensing data), your approach,
and the main result. -->

## Repository structure

```
.
├── Code/
│   ├── configs/                 # YAML configs (default.yaml + one per experiment)
│   ├── src/action_decisions/    # Python package
│   │   ├── data/                # dataset, modality readers, transforms, splits
│   │   ├── models/
│   │   ├── training/            # trainer, losses, checkpointing
│   │   ├── evaluation/          # metrics
│   │   └── utils/               # seeding, logging, config loading
│   ├── scripts/                 # entry points: preprocess, train, evaluate, make_figures
│   ├── notebooks/               # exploration
│   ├── tests/
│   └── requirements.txt
├── Data/                        # DeepSense 6G scenarios 31–35 (not tracked in git, see below)
├── Visuals/                     # Figures and plots used in the thesis
└── Readme.md
```

## Data

The experiments use the [DeepSense 6G](https://www.deepsense6g.net/) dataset, **Scenarios 31–35**.
These are real-world vehicle-to-infrastructure (V2I) recordings with a 60 GHz mmWave link. Unit 1 is a
stationary basestation and Unit 2 is a moving vehicle.

| Scenario | Samples | Size on disk | Index file |
|----------|--------:|-------------:|------------|
| 31       | 7,012   | ~22 GB       | `scenario31_dev.csv` |
| 32       | 3,235   | ~10 GB       | `scenario32_dev.csv` |
| 33       | 3,981   | ~13 GB       | `scenario33_dev.csv` |
| 34       | 4,439   | ~14 GB       | `scenario34.csv` |
| 35       | 3,045   | ~11 GB       | `scenario35.csv` |

### Scenarios 31–34

Each sample has these modalities:

| Modality | Unit | Location | Format |
|----------|------|----------|--------|
| RGB camera | 1 | `unit1/camera_data/` | `.jpg` |
| mmWave received power (64 beams) | 1 | `unit1/mmWave_data/` | `.txt` |
| LiDAR point cloud | 1 | `unit1/lidar_data/` | `.ply` |
| FMCW radar | 1 | `unit1/radar_data/` | `.npy` |
| Basestation GPS (fixed) | 1 | `unit1/GPS_data/gps_location.txt` | `.txt` |
| Vehicle GPS | 2 | `unit2/GPS_data/` | `.txt` |

The index CSV maps each sample to its files. It also holds the labels and metadata: the optimal beam
index (`unit1_beam`), the max received power, the timestamp, the sequence index, and the vehicle's GPS
speed and fix quality.

### Scenario 35

Scenario 35 is laid out differently:

- It has **no LiDAR** data.
- The folder names are different: `unit1/{rgb,pwr,radar,gps}` and `unit2/gps`.
- The columns are named differently: `best_beam` and `max_power` instead of `unit1_beam` and
  `unit1_max_pwr`, and `abs_idx` instead of `index`.
- It adds radar labels: `radar_label_bool`, plus `radar_label`, which points to a file in
  `resources/radar_label/`.
- There are fewer power, radar and GPS files than rows, so some samples reuse a file. Always resolve
  paths through the CSV rather than by filename.

### Getting the data

The data is too large for git and is excluded via `.gitignore`. To reproduce:

1. Download Scenarios 31–35 from the [DeepSense 6G website](https://www.deepsense6g.net/).
2. Extract them into `Data/` so the layout is:
   ```
   Data/scenario-31/scenario31_dev.csv
   Data/scenario-32/scenario32_dev.csv
   Data/scenario-33/scenario33_dev.csv
   Data/scenario-34/scenario34.csv
   Data/scenario-35/scenario35.csv
   ```

## Setup

<!-- Framework not chosen yet; fill in once Code/requirements.txt has content. -->
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r Code/requirements.txt
```

## Usage

<!-- One command per pipeline stage, e.g.:
python Code/scripts/preprocess.py --config Code/configs/default.yaml
python Code/scripts/train.py --config Code/configs/experiments/<name>.yaml
python Code/scripts/evaluate.py --checkpoint Code/outputs/<run>/best.pt
python Code/scripts/make_figures.py      # writes to Visuals/
-->

## Results

<!-- Main table/figure from the thesis, with links to Visuals/ -->

## Citation

If you use this work, please cite the thesis:

```bibtex
@mastersthesis{saffarmoghadam<year>,
  author = {Elias Saffarmoghadam},
  title  = {<title>},
  school = {<university>},
  year   = {<year>}
}
```

and the DeepSense 6G dataset:

```bibtex
@article{alkhateeb2023deepsense,
  author  = {Alkhateeb, Ahmed and Charan, Gouranga and Osman, Tawfik and Hredzak, Andrew and Morais, Jo{\~a}o and Demirhan, Umut and Srinivas, Nikhil},
  title   = {{DeepSense 6G}: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset},
  journal = {IEEE Communications Magazine},
  year    = {2023}
}
```
