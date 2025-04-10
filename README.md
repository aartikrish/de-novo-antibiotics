# A Generative Deep Learning Approach to _de novo_ Antibiotic Design

This repository accompanies the manuscript **"A generative deep learning approach to _de novo_ antibiotic design"** by Krishnan, Anahtar, Valerie, et al.

---

## 📁 Table of Contents

- [1. Setting up the Environment](#1-setting-up-the-environment)
- [2. `models/` – Chemprop Scoring](#2-models--chemprop-scoring)
- [3. `downselection/` – Filtering Hits for Fragment-Based Generation](#3-downselection--filtering-hits-for-fragment-based-generation)
- [4. `fragment_de_novo/` – Fragment-Based Generation (F1, F2, F2′)](#4-fragment_de_novo--fragment-based-generation-f1-f2-f2)
- [5. `entirely_de_novo/` – Full Molecule Generation](#5-entirely_de_novo--full-molecule-generation)
  
---

## 1. Setting up the Environment

We recommend using Conda to manage environments. Clone this repository and install dependencies as follows:

```bash
conda create -n antibiotics python=3.8
conda activate antibiotics
conda install -c conda-forge rdkit
pip install git+https://github.com/bp-kelley/descriptastorus
pip install chemprop
pip install pandas numpy tqdm os subprocess tqdm sys time crem 
```

---

## 2. `models/` – Chemprop Scoring

This work uses Chemprop ensemble models trained to predict antibiotic activity against _Staphylococcus aureus_ and _Neisseria gonorrhoeae_. Additionally, we provide three Chemprop models for cytotoxicity prediction against HepG2, IMR90, and HSkMC human cell lines.

### Train a Chemprop Model

```bash
chemprop_train \
  --data_path models/example/data/data_file.csv \
  --dataset_type classification \
  --config_path hyperparameters.json \
  --save_dir checkpoints/ \
  --ensemble_size 20 \
  --features_generator rdkit_2d_normalized \
  --no_features_scaling
```

### Predict with a Pretrained Chemprop Model

```bash
chemprop_predict \
  --test_path fragments.csv \
  --checkpoint_dir models/example/checkpoints/ \
  --preds_path fragment_preds.csv \
  --features_generator rdkit_2d_normalized \
  --no_features_scaling
```

---

## 3. `downselection/` – Filtering Hits for Fragment-Based Generation

We provide Jupyter notebooks to downselect predicted hits for _S. aureus_ and _N. gonorrhoeae_ separately.

### Downselection Criteria

1. Filter based on Chemprop-predicted antibiotic activity.
2. Remove fragments with high predicted cytotoxicity.
3. Apply PAINS and Brenk structural filters.
4. Remove compounds similar to known antibiotics.
5. (Optional) Manual inspection.

Final filtered CSVs are included in this folder.

---

## 4. `fragment_de_novo/` – Fragment-Based Generation (F1, F2, F2′)

This folder includes generative runs seeded on three fragments: **F1, F2, and F2′**. Each was expanded using both CReM and F-VAE. The code used for molecule generation with F-VAE can be found at the original [GitHub](https://github.com/wengong-jin/multiobj-rationale). The code for generating molecules with F-CReM can be found in `crem_molecule_generation/`. 

### Results Summary

| Fragment | Method   | Generated   | Downselected |
|----------|----------|-------------|--------------|
| F1       | F-CReM   | 518,203     | 285          |
| F1       | F-VAE    | 6,937,677   | 678          |
| F2       | F-CReM   | 141,109     | –            |
| F2       | F-VAE    | 3,297,292   | –            |
| F2′      | F-CReM   | 106,557     | –            |
| F2′      | F-VAE    | 18,164,746  | 9,684        |

---

## 5. `entirely_de_novo/` – Full Molecule Generation

We also explored _de novo_ generation from scratch using:

- **JT-VAE**: A variational autoencoder that generates molecules from latent space. 
- **CReM**: Random fragment recombination starting from seed three molecules (H₂O, NH₃, CH₄).

The code used for de novo molecule generation with JT-VAE can be found at the original [GitHub](https://github.com/wengong-jin/hgraph2graph). The code for generating molecules de novo with ReM can can be found in `crem_molecule_generation/`. 

### Results Summary

| Method  | Total Molecules |
|---------|------------------|
| JT-VAE  | 4,831            |
| CReM    | 55,031           |

All generated molecules were scored on-the-fly using Chemprop and then filtered as described in the paper. 
