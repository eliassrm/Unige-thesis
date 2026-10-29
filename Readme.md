# <Thesis title>

Code, data, and figures for my thesis, *<full title>* (<degree>, <university>, <year>).

<!-- 2–4 sentences: the problem, your approach, and the main result. For example:
"This thesis looks at predicting the optimal mmWave beam for a moving vehicle from side information
(camera, LiDAR, radar, GPS) instead of exhaustive beam sweeping. ..." -->

## Repository structure

```
.
├── Code/        # Source code (preprocessing, models, training, evaluation)
├── Data/        # DeepSense 6G scenarios 31–33 (not tracked in git, see below)
├── Visuals/     # Figures and plots used in the thesis
└── Readme.md
```

## Data

The experiments use the [DeepSense 6G](https://www.deepsense6g.net/) dataset, **Scenarios 31, 32, and 33**.
These are real-world vehicle-to-infrastructure (V2I) recordings with a 60 GHz mmWave link. Unit 1 is a
stationary basestation and Unit 2 is a moving vehicle.

| Scenario | Samples | Size on disk |
|----------|--------:|-------------:|
| 31       | 7,012   | ~22 GB       |
| 32       | 3,235   | ~10 GB       |
| 33       | 3,981   | ~13 GB       |

Each sample has these modalities:

| Modality | Unit | Location | Format |
|----------|------|----------|--------|
| RGB camera | 1 | `unit1/camera_data/` | `.jpg` |
| mmWave received power (64 beams) | 1 | `unit1/mmWave_data/` | `.txt` |
| LiDAR point cloud | 1 | `unit1/lidar_data/` | `.ply` |
| FMCW radar | 1 | `unit1/radar_data/` | `.npy` |
| Basestation GPS (fixed) | 1 | `unit1/GPS_data/gps_location.txt` | `.txt` |
| Vehicle GPS | 2 | `unit2/GPS_data/` | `.txt` |

`scenarioXX_dev.csv` maps each sample index to its files. It also holds the labels and metadata: the
optimal beam index (`unit1_beam`), the max received power, the timestamp, the sequence index, and the
vehicle's GPS speed and fix quality.

### Getting the data

The data is too large for git and is excluded via `.gitignore`. To reproduce:

1. Download Scenarios 31, 32 and 33 from the [DeepSense 6G website](https://www.deepsense6g.net/).
2. Extract them into `Data/` so the layout is:
   ```
   Data/scenario-31/scenario31_dev.csv
   Data/scenario-32/scenario32_dev.csv
   Data/scenario33/scenario33_dev.csv
   ```

## Setup

<!-- Fill in once the code exists, e.g.: -->
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

<!-- One command per pipeline stage, e.g.:
python Code/preprocess.py --scenario 31
python Code/train.py --modality camera+gps
python Code/evaluate.py --checkpoint ...
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
