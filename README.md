# MMPL
Momentum Morphological Prototype Learning for Patch Search in Whole Slide Images

## Repository structure

```
.
├── config/
│   └── camelyon16_r50.yaml         # Experiment config (OmegaConf YAML)
├── scripts/
│   └── run_mil_camelyon.sh         # Convenience launcher
├── sample_data/
│   └── CAMELYON16/                 # Sample label / split files (format reference)
│       ├── CAMELYON16_all_labels.csv
│       └── CAMELYON16_split_train_val_test.csv
├── src/
│   ├── dataset.py                  # Bag dataset (.h5 features or .pkl raw JPEG bytes)
│   ├── modeling.py                 # MMPL model, encoders, Sinkhorn-Knopp
│   ├── training.py                 # Trainer (train / validate / test loops)
│   ├── utils.py                    # Metrics, logging, checkpoint saver
│   └── main.py                     # Entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n mmpl python=3.9 -y
conda activate mmpl

# Install PyTorch built for your CUDA version (CUDA 12.1 shown here):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

> `PyTurboJPEG` is only required when training end-to-end from raw patch bytes
> (`.pkl` inputs). It depends on the system `libturbojpeg` library
> (`apt-get install libturbojpeg`). It can be skipped when using precomputed features.

## Data

### Preprocessing

Patch extraction and feature embedding follow the **ZoomMIL** preprocessing pipeline:
<https://github.com/histocartography/zoommil>. We use it to tessellate each WSI into
patches and embed them with a ResNet50 encoder, storing the resulting feature vectors
per slide.

Alternatively, the widely used **CLAM** toolkit (<https://github.com/mahmoodlab/CLAM>)
can be used for patch extraction and feature embedding; just make sure the resulting
features are saved per slide under the dataset column expected by the config
(`data.dataset_column`, default `20.0x_patches`).

### Input format

Each slide is a *bag* of patches. Two input formats are supported:

- **`.h5`** — precomputed patch feature vectors stored under the dataset column
  (default `20.0x_patches`). This is the default mode (`model.backbone: "None"`).
- **`.pkl`** — raw JPEG bytes of the patches, for end-to-end training with a learnable
  vision encoder (`model.backbone: "ResNet50"` or `"UNI"`).

All slides live in a single directory, one file per slide (`<slide_id>.h5` for
precomputed features, or `<slide_id>.pkl` for raw JPEG patches). Two CSV files describe
the labels and the splits:

- `*_all_labels.csv` — indexed by `slide_id`, with a `type` column holding the class id.
- `*_split_train_val_test.csv` — `train` / `val` / `test` columns listing slide names.

Sample files in this exact format are provided under
[`sample_data/CAMELYON16/`](sample_data/CAMELYON16) for reference:

```csv
# CAMELYON16_all_labels.csv
slide_id,type
normal_001,0
tumor_001,1
...
```

```csv
# CAMELYON16_split_train_val_test.csv
train,val,test
normal_001,normal_002,normal_003
tumor_001,tumor_002,tumor_003
```

> The sample directory ships only the label/split CSVs as a format reference; the
> corresponding `.h5` feature files are not included. Generate them with the ZoomMIL
> pipeline above and place them in the directory pointed to by `data.directory`.

Point the config at your files:

```yaml
data:
  directory: ./data/CAMELYON16/features-r50
  dataset_splits: ./data/CAMELYON16/CAMELYON16_split_train_val_test.csv
  dataset_labels: ./data/CAMELYON16/CAMELYON16_all_labels.csv
  dataset_column: 20.0x_patches
```

### End-to-end training (raw JPEG patches)

The default mode (`model.backbone: "None"`) consumes **precomputed** feature vectors and
keeps the vision encoder frozen. To instead train the vision encoder **end to end**
(`model.backbone: "ResNet50"` or `"UNI"`), the patches must be stored as **raw JPEG
bytes** rather than features, so the encoder can decode and embed them on the fly. This
lets the encoder be trained while only the retrieved patches are decoded each step,
avoiding the cost of processing every patch in a slide.

In this case, save the JPEG-encoded patches per slide as `<slide_id>.pkl` (keyed by
`data.dataset_column`). At runtime these bytes are decoded with TurboJPEG (hence the
`PyTurboJPEG` dependency) and embedded by the vision encoder. Point `data.directory` at
the folder of `.pkl` files and set `model.backbone` accordingly.

## Training

Run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python src/main.py config/camelyon16_r50.yaml experiment.seed=0
```

Or use the launcher script (which accepts `GPU`, `SEEDS`, and `CONFIG` env vars):

```bash
GPU=0 SEEDS="0 1" bash scripts/run_mil_camelyon.sh
```

Any config field can be overridden from the command line via OmegaConf dotlist syntax:

```bash
python src/main.py config/camelyon16_r50.yaml \
    model.num_prototypes=15 model.retrieve_k=100 experiment.seed=0
```

Training runs validation each epoch, saves the best checkpoint (by
`experiment.save_metric`) plus the latest one to `experiment.directory`, and finally
evaluates on the test set, writing the scores to a `*_test_score.json` file.

### Key configuration options

| Field | Description |
| --- | --- |
| `model.backbone` | `"None"` (precomputed features), `"ResNet50"`, or `"UNI"` |
| `model.num_prototypes` | Number of learnable prototypes |
| `model.retrieve_k` | Total retrieval budget (patches selected per slide) |
| `model.queue_size` | Size of the feature queue for Sinkhorn-Knopp |
| `model.sinkhorn_weight` | Weight of the Sinkhorn pseudo-label loss |
| `model.temperature` | Temperature for the prototype assignment loss |
| `experiment.resume` | Resume from a checkpoint in `experiment.directory` |

### Experiment tracking

Weights & Biases logging is disabled by default (`wandb.use: false`). Set it to `true`
in the config (and `wandb login`) to enable it.

## Notes

- The `UNI` backbone downloads gated weights from Hugging Face. Authenticate first via
  `huggingface-cli login` or by exporting `HF_TOKEN` before running.