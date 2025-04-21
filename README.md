# A Generative Deep Learning Approach to _de novo_ Antibiotic Design

This repository accompanies the manuscript **"A generative deep learning approach to _de novo_ antibiotic design"** by Krishnan, Anahtar, Valerie, et al., 2025. The datasets accompanying it can be found on [Zenodo](https://doi.org/10.5281/zenodo.15191826).


# Summary
The antimicrobial resistance crisis necessitates structurally novel antibiotics. While deep learning approaches can identify antibacterial compounds from existing libraries, structural novelty remains limited. Here, we developed a generative artificial intelligence framework for designing _de novo_ antibiotics through two approaches: (1) a fragment-based method to comprehensively screen >10^7 chemical fragments in silico against _Neisseria gonorrhoeae_ or _Staphylococcus aureus_, subsequently expanding promising fragments, and (2) unconstrained de novo compound generation, each using genetic algorithms and variational encoders. Of 24 synthesized compounds, seven demonstrated selective antibacterial activity. Two lead compounds exhibited bactericidal efficacy against multidrug-resistant isolates, distinct mechanisms of action, and reduced bacterial burden in vivo in mouse models of _N. gonorrhoeae_ vaginal infection and methicillin-resistant _S. aureus_ skin infection. We further validated structural analogs for both compound classes as antibacterial. Our approach enables the generative deep learning-guided design of de novo antibiotics, providing a platform for mapping uncharted regions of chemical space.

---

## 📁 Table of Contents

- [1. Setting up the Environment](#1-setting-up-the-environment)
- [2. Chemprop Models](#2-chemprop-models)
- [3. Filtering Molecules – `downselection/`](#3-filtering-molecules)
- [4. Fragment-Based Molecule Generation – `crem_molecule_generation/`](#4-fragment-based-molecule-generation)
- [5. Full Molecule Generation](#5-full-molecule-generation)
- [6. tSNE Plot – `tSNE_plot/`](#6-tsne-plot)
  
---

## 1. Setting up the Environment

We recommend using Conda to manage environments. Clone this repository and install dependencies as follows:

```bash
conda create -n antibiotics python=3.8
conda activate antibiotics
conda install -c conda-forge rdkit 
pip install git+https://github.com/bp-kelley/descriptastorus
pip install chemprop==1.7.1
pip install -r requirements.txt
```

Note that this environment is incompatible with the exact package versions needed to predict RA_scores, see also the original [GitHub](https://github.com/reymond-group/RAscore). It is best to create a separate conda environment for this 

```bash
conda create -n ra_score python=3.7
conda activate ra_score
conda install -c rdkit rdkit -y
pip install git+https://github.com/reymond-group/RAscore.git@master
```
---

## 2. Chemprop Models 

This work uses Chemprop ensemble models trained to predict antibiotic activity against _Staphylococcus aureus_ and _Neisseria gonorrhoeae_. Additionally, we provide three Chemprop models for cytotoxicity prediction against HepG2, IMR90, and HSkMC human cell lines. The checkpoints and training can be found at [Zenodo](https://doi.org/10.5281/zenodo.15191826).

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

## 3. Filtering Molecules – `downselection/`

We provide a general Jupyter notebooks to downselect molecules based on the filters used in the manuscript.  

### Downselection Criteria

* Antibiotic activity based on Chemprop scores (Staph. aureus, N. gonorrhoeae): ```ACTIVITY_THRESHOLD```
* Cytotoxicity based on Chemprop scores (IMR90, HSkMC, HepG2): ```TOXICITY_THRESHOLD```
* PAINS and Brenk filters (binary, no threshold needed)
* Tanimoto similarity to 559 known antibiotics: ```TANSIM_THRESHOLD```
* Retrosynthetic accessbility (SA or RA) score: ```SA_RA_THRESHOLD```

The filtered CSVs and the original SMILES lists can be found at [Zenodo](https://doi.org/10.5281/zenodo.15191826).

---

## 4. Fragment-Based Molecule Generation (F2, F3, F3′) – `crem_molecule_generation/` 

This folder includes generative runs seeded on three fragments: **F2, F3, and F3′**. Each was expanded using both CReM and F-VAE. The code used for molecule generation with F-VAE can be found at the original [GitHub](https://github.com/wengong-jin/multiobj-rationale). The code for generating molecules with CReM can be found in `crem_molecule_generation/` and is adapted by simply changing the SMILES of input fragment. 

The list of generated molecules can be found at [Zenodo](https://doi.org/10.5281/zenodo.15191826).

### Results Summary

| Fragment | Method   | Generated   | Downselected |
|----------|----------|-------------|--------------|
| F2       | F-CReM   | 518,203     | 285          |
| F2       | F-VAE    | 6,937,677   | 678          |
| F3       | F-CReM   | 141,109     | –            |
| F3       | F-VAE    | 3,297,292   | –            |
| F3′      | F-CReM   | 106,557     | –            |
| F3′      | F-VAE    | 18,164,746  | 9,684        |

---

## 5. Full Molecule Generation

We also explored _de novo_ generation from scratch using:

- **JT-VAE**: A variational autoencoder that generates molecules from latent space. 
- **CReM**: Random fragment recombination starting from seed three molecules (H₂O, NH₃, CH₄).

The code used for de novo molecule generation with JT-VAE can be found at the original [GitHub](https://github.com/wengong-jin/hgraph2graph). The same code for generating molecules from fragments with CReM was used to generate de novo CReM molecules. Simply adapt the code found in `crem_molecule_generation/` by using H₂O, NH₃, CH₄ as input fragments. 

### Results Summary

| Method  | Generated  | Downselected |
|---------|------------|--------------|
| JT-VAE  | 28,534,490 | 4,831        |
| CReM    | 480,484    | -            |

All generated molecules were scored on-the-fly using Chemprop and then filtered as described in the paper. 

## 6. tSNE Plot – `tSNE_plot/` 

The code to generate the tSNE plot is given. The files with the SMILES list used as input can be found on [Zenodo](https://doi.org/10.5281/zenodo.15191826).
