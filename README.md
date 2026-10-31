In the name of God

# Spatial Transcriptomic Prediction in Cancer Pathology under Missing Modalities Using Pathology Foundation Models

A self-contained benchmark for predicting spot-level gene expression from routine H&E histology when parts of the input
are **missing**. It contains our model (**OURS**), several variants of it, and re-implementations of published
spatial-transcriptomics predictors that are run under exactly the same data, missing-data and evaluation protocol.

---

## 1. Task

For every Visium spot we predict the expression vector `y ∈ R^50` (log(1 + counts-per-10⁴) of the 50 genes with the
largest variance, selected on the **training patients only**) from three kinds of input:

| input | content |
|---|---|
| **H&E image** | the 224 × 224 px patch of the spot |
| **Nuclei** | five CellViT cell-type counts of the spot (neoplastic, inflammatory, connective, epithelial, dead) |
| **Context** | the H&E patches of up to six neighbouring spots (hexagonal grid) |

Any of the three can be absent for a spot (simulated at chosen percentages, or absent for physical reasons such as
spots without neighbours, or slides with incomplete nuclei segmentation). Train, validation and test sets are split at
the **patient level** (provided by the dataset reader).

## 2. Methods

### 2.1 Our model (OURS)

1. **Frozen foundation-model embeddings** (computed once and cached) for the spot patch and for each neighbouring patch;
   the context vector is the mean of the neighbour embeddings.
2. **Nuclei branch**: `log(1+n)` of the five counts and the five fractions `n_c / (1 + Σn)`, standardised with training
   statistics, followed by a small network (10 → 64 → 128).
3. **Projection**: each of the three vectors is mapped to 512 dimensions (Linear → LayerNorm → GELU).
4. **Fusion** (pool, see §2.3): by default concatenation (1536-d).
5. **Prediction head**: LayerNorm → Linear(1024) → GELU → Dropout(0.2) → Linear(50).
6. **Training**: MSE on z-scored genes, AdamW (lr 3e-4, weight decay 0.05), batch 256, ≤ 60 epochs, cosine schedule with
   2 warm-up epochs, early stopping (patience 10) on validation mean per-gene PCC; the best validation epoch is restored.
   Final prediction = mean of an ensemble of five models (different seeds).

All of these numbers live in `main.py` (`HPARAMS`, `ENSEMBLE_SIZE_*`) and can be changed.

### 2.2 OURS

| name | description |
|---|---|
| `OURS-0`, `OURS-Mean` | the model above, with the two ways of handling absent inputs (§3) |


### 2.3 Pools of encoders and fusions

OURS-family methods are run for every combination of

* **`FOUNDATION_ENCODERS`** (frozen, cached): `hoptimus0`, `hoptimus1`, `uni`, `gigapath` (pathology foundation models,
  gated on Hugging Face), `resnet50`, `densenet121`, `vit_b16`, `dinov2` (generic). Aliases `optimus0` / `optimus1` work.
* **`FUSIONS`**: `concat` (simple concatenation), `sum` (mean), `gate` (learned per-spot weights over the three inputs),
  `attn` (a transformer layer over three tokens), `lmf` (low-rank multiplicative fusion).

Result names look like `OURS-0-hoptimus1-concat`. The **first** encoder and fusion in the lists are the *primary* ones:
they are used by the competitors' `Our-Encoder` variants and give the reference method of the Wilcoxon tests
(`OURS-0-<first encoder>-<first fusion>`).

### 2.4 Competitors (own re-implementations, nothing imported from the original repositories)

| method | what is implemented | adaptation to this task |
|---|---|---|
| **BLEEP** | contrastive image ↔ expression embedding, prediction by retrieving the expression of the top-k most similar training spots | expression encoder takes the 50 z-scored genes |
| **mclSTExp** | contrastive learning where the expression branch adds x/y position embeddings and self-attention over the spots of the batch; same retrieval inference | positions are normalised per slide |
| **GHIST** | multitask network: expression from softmax-weighted learnable cell-type profiles plus an offset, with an auxiliary loss on the neighbourhood cell-type composition | **spot-level adaptation**: no cell segmentation or single-cell resolution; the CellViT nuclei composition is the auxiliary target |
| **STORM** | low-rank factorisation of expression with spatial smoothness, image alignment and gene-graph regularisers, adaptive loss weights | **inductive adaptation**: free latents for training spots, test spots predicted from features followed by spatial smoothing; the PPI graph is replaced by a training co-expression graph (STRING is not available offline) |
| **MIST** | outside the scope of this benchmark | it addresses a different data type (single-T-cell RNA and TCR sequences), whereas this benchmark works with H&E, nuclei and context |

Every competitor exists in two **variants**:

* `Default` – the competitor's own image encoder (ResNet50 for BLEEP, DenseNet121 for mclSTExp, a UNet for GHIST,
  frozen ResNet50 features for STORM).
  The same encoder also processes up to `DEFAULT_CTX_PATCHES` neighbour patches and a small network processes the nuclei, so
  that every method sees all three inputs. Pixels are down-scaled to `NATIVE_IMG_SIZE`.
* `Our-Encoder` – the encoder of OURS (frozen embeddings + nuclei network + projections + fusion) replaces the image
  encoder; the competitor's remaining design (heads, graphs, losses) is unchanged.

and with both ways of handling absent inputs (`0`, `Mean`), e.g. `BLEEP-Mean-Our-Encoder`.

## 3. Handling of absent inputs

Absent inputs are **imputed**; no indicator of absence is given to any model, and the rule is identical for all methods.

* **`0`** – absent H&E image / context → zero embedding (pixel models: black patch); absent nuclei → zero counts, then
  the usual transformation and standardisation.
* **`Mean`** – replaced by the average over the **training** spots in which that input is present (embedding mean,
  mean patch, mean standardised nuclei features).

Statistics are computed on training spots only, and values of absent inputs are never read.

Missing-data **configurations** are strings such as `'100-image-20-nuclei-20-context'` (percentage of spots keeping each
input) or the joint pattern `'complex-30-30-5-5-20-5-5'`. Run *r* uses seed `BASE_SEED + r` for both the removal pattern
and the model initialisation, so all methods see the same pattern within a run.

## 4. Evaluation

Computed on the test patients, on the original log scale, per run (ensemble prediction):

| metric | definition | better |
|---|---|---|
| **MSE**, **MAE** | over all test spots × genes | lower |
| **PCC** | per test slide and gene Pearson correlation over spots, averaged over genes then slides | higher |
| **SCC** | same with Spearman rank correlation | higher |
| **R²** | `1 − SSE/SST` per gene with SST around the *training* mean (R² > 0 ⇔ better than predicting the training mean), averaged over genes | higher |

The end of the run prints, for every configuration, mean ± std over runs with the **winner** of each metric, a
**Wilcoxon signed-rank test** of the reference method against every other method, and an overall average-rank winner.

> **About the Wilcoxon test.** With only a few runs, a test across runs has limited resolution (the smallest possible p-value
> is 0.25 for 3 runs), so the test is paired over the 50 **genes** (each averaged over runs). It summarises how consistently
> the reference method is ahead across genes on the fixed test patients; the variation between runs is given by the
> standard deviations in the tables, and broader statements about new patients are best supported by cross-validation over
> patients. With many methods, a multiple-comparison correction (Holm/Bonferroni) can be applied to the p-values.

## 5. Installation

```bash
pip install torch torchvision timm huggingface_hub h5py anndata numpy scipy
```

Python ≥ 3.9. A GPU is strongly recommended (the `Default` pixel-trained competitors are the expensive part).

Required next to `main.py`:

* the dataset reader `Hest_Reg_Lymph_INT1to24.py` (or `hest_reg_lymph_int1to24.py`);
Please note that we have three datasets. To apply, you should create a folder for each one you want, then download it using the provided donwloader code: 
```bash
python download.py
```
then copy other provide py files into that folder. then run main.py. Please consider that for simplisity we used `Hest_Reg_Lymph_INT1to24.py` to read each of these dataset, but for each dataset, just one line should be changed, we have already done, so, you should pay attention that `Hest_Reg_Lymph_INT1to24.py` in folder dataset 1,2,3 are a different in just one line.
* the data folder (default `hest_data/`) with `patches/<ID>.h5` and `splits_INT1to24.json`, as expected by the reader.

**Gated models.** `hoptimus0/1`, `uni`, `gigapath` need an approved access request on their Hugging Face pages for the
account whose token is used. Encoders that cannot be loaded are skipped with a warning (the first encoder of the list is required).
The repository does not contain a token: `HF_TOKEN = '****'` is a neutral value (the login is skipped for it), and the real
token is set locally.

## 6. Usage

Edit the settings block at the top of `main.py`, then

```bash
python main.py
```

Main settings:

| setting | meaning |
|---|---|
| `CONFIGS` | missing-data configurations to run |
| `N_RUNS`, `BASE_SEED` | number of repetitions and first seed |
| `COMPETITORS` | e.g. `[("OURS","0","mean"), ("BLEEP","0")]`; methods: `OURS`, `BLEEP`, `GHIST`, `MCLSTEXP`, `STORM`, `FMH2ST` |
| `VARIANTS` | `("default","our_encoder")` – which encoder variants of the competitors to run (no effect on OURS-family) |
| `FOUNDATION_ENCODERS`, `FUSIONS` | the pools of §2 |
| `HPARAMS`, `RAW_EPOCHS`, `RAW_PATIENCE`, `RAW_MAX_STEPS_PER_EPOCH` | optimisation settings |
| `ENSEMBLE_SIZE_CACHED`, `ENSEMBLE_SIZE_RAW` | models averaged for cheap (cached-feature) / expensive (pixel) methods |
| `NATIVE_IMG_SIZE`, `DEFAULT_CTX_PATCHES` | resolution and number of neighbour patches of `Default` pixel encoders |
| `OUT_DIR`, `DEVICE` | output folder, device |

The number of trainings is (methods) × (imputations) × (encoders × fusions for the OURS family) × (variants for competitors)
× runs × configs × ensemble size. The lists can be shortened to reduce the cost; the shipped defaults cover the full set of encoders and fusions.

Embeddings and down-scaled patches are cached in `<ROOT>/_cache_mm_bench/` (one file per encoder and slide), so repeated
runs and new encoders only compute what is missing.

## 7. Outputs

| file | content |
|---|---|
| `output-MSE.txt`, `output-MAE.txt`, `output-PCC.txt`, `output-SCC.txt`, `output-R2.txt` | per configuration, per method, one value per run (one per line) |
| `output-gene-<METRIC>.csv` | per-gene values of every run (`config, method, run, gene_index, value`) for offline statistics |
| `output-summary.txt` | tables with mean ± std, winners, Wilcoxon tests, average ranks |

Files are rewritten after every fitted method, so results survive a crash. Every output file starts with *In the name of God*.

## 8. Project structure

```
main.py          settings, experiment loop, method list expansion
data_utils.py    dataset building, missing-data simulation, caches, imputation, batching, spatial windows
encoders.py      OURS encoder with fusion pool, competitors' pixel encoders
methods.py       OURS family and competitor re-implementations, training loop
reporting.py     metrics, Wilcoxon tests, winners, output files
test/            synthetic smoke test (generated slides; checks that the whole pipeline runs end to end)
```

## 9. Notes

* The competing methods are the original published methods adapted to handle this spot-level, multi-input, patient-split
  regression task (details in §2.4); results are therefore reported for these adapted versions.
* `Default` competitors trained from pixels use down-scaled patches and a reduced number of epochs for tractability; both are
  configurable (`NATIVE_IMG_SIZE`, `RAW_EPOCHS`, `RAW_MAX_STEPS_PER_EPOCH`).
* As a numerical safeguard, non-finite predictions (if any) are replaced by the training mean and a warning is printed.
* The statistical tests are paired over genes (§4); for statements about new patients, evaluation on more test patients
  (e.g. cross-validation over patients) is recommended.

## 10. Acknowledgements

This work builds on the methods BLEEP, GHIST, mclSTExp and STORM, on the single-cell nuclei annotations of CellViT,
on the pathology foundation models H-optimus-0/1, UNI and Prov-GigaPath, and on the HEST-1k dataset; please cite the
original works accordingly.
