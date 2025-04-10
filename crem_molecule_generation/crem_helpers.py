"""Helper function for CReM analysis."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob as glob
import subprocess
import sys
import os

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import RDConfig

from filters import process_molset

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa


def calculateScoreThruToxModel(
    patterns, results_folder_name, results_file_name, tox_model
):
    """
    Function to calculate scores through a toxicity model.

    This function generates a clean CSV file,
    runs a command line through Jupyter using subprocess,
    and returns a list of SMILES and corresponding toxicity scores.

    :param patterns: List of patterns
    :param results_folder_name: Name of the results folder
    :param results_file_name: Name of the results file
    :param tox_model: Type of toxicity model ('primary' or 'hepg2')
    :return: List containing SMILES and corresponding toxicity scores
    """
    # write to clean csv
    clean_name = results_folder_name + tox_model + "_" + results_file_name
    new_df = pd.DataFrame(patterns, columns=["SMILES"])
    new_df.to_csv(clean_name, index=False)

    # use subprocess to run command line thru jupyter notebook
    # could easily just run command line but this is automated
    activate_command = "source activate chemprop; "
    if tox_model == "primary":
        model_folder = "final_tox_primary"
    elif tox_model == "hepg2":
        model_folder = "final_tox_hepg2"
    run_command = (
        "python predict.py --test_path "
        + "../generativeML/out/"
        + clean_name
        + " --checkpoint_dir ../generativeML/models/"
        + model_folder
        + " --preds_path "
        + "../generativeML/out/"
        + clean_name
        + " --features_generator rdkit_2d_normalized "
        + "--no_features_scaling --smiles_columns SMILES"
    )
    full_command = activate_command + run_command
    subprocess.run(
        full_command,
        cwd="../../chemprop/",
        shell=True,
        capture_output=True)
    preds = pd.read_csv(clean_name)

    new_smis = list(preds["SMILES"])
    new_scores = list(preds["TOXICITY"])
    if len(new_smis) > 0:
        return [new_smis, new_scores]
    else:
        return []


def collate_crem_molecules_from_multiple_rounds(
    results_path, out_dir, smi_col="SMILES", hit_col="ACTIVITY"
):
    """
    Collates molecules from multiple rounds of results.

    Combine pandas DataFrames.

    :param results_path: List of paths to results
    :param out_dir: Output directory
    :param smi_col: Name of the SMILES column (default: 'SMILES')
    :param hit_col: Name of the hit column (default: 'ACTIVITY')
    :return: DataFrame containing collated molecules
    """
    columns = [
        "Score",
        "Grow_or_Mut",
        "Algorithm_Params",
        "Round",
        smi_col,
        hit_col]
    allmolsdf = pd.DataFrame(columns=columns)
    currmolsdf = pd.DataFrame(columns=columns)

    for path in results_path:
        allmolsdf = pd.concat([allmolsdf, currmolsdf])
        currmolsdf = pd.DataFrame(columns=columns)

        for filename in glob.glob(out_dir + path + "/*/*_scores.csv"):
            if "modified_score" in filename:
                method = "modified_score"
            else:
                method = "chemprop_score"
            if "grow" in filename:
                mod = "grow"
            elif "mutate" in filename:
                mod = "mutate"

            param_names = filename.split(mod + "/")[1].split("/")[0]
            rd = filename.split("all_mols_round_")[1].split("_scores.csv")[0]
            df = pd.read_csv(filename)

            try:
                _ = df[
                    df[hit_col] > 0.1
                ]  # some did not make it to have their activities calculated
            except Exception as e:
                print(e)
                continue
            df["Score"] = [method] * len(df)
            df["Grow_or_Mut"] = [mod] * len(df)
            df["Algorithm_Params"] = [param_names] * len(df)
            df["Round"] = ["Round " + str(rd)] * len(df)
            currmolsdf = pd.concat([currmolsdf, df])
    allmolsdf = pd.concat([allmolsdf, currmolsdf])

    return allmolsdf


def hist_plot(data, xs, xlabel, hue="Round"):
    """
    Plot histogram.

    :param data: DataFrame containing data
    :param xs: x-axis values
    :param xlabel: Label for x-axis
    :param hue: Hue variable (default: 'Round')
    """
    plt.figure(figsize=(3, 3), dpi=300)
    sns.histplot(data=data, x=xs, hue=hue)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def filter_crem_dataframe(
    allmolsdf,
    smi_col,
    hit_col,
    out_dir,
    hit_thresh=0.5,
    sascore_thresh=3,
    tan_to_abx=0.5,
    abx_path="",
    abx_smiles_col="SMILES",
    hepg2_tox_thresh=0.2,
    prim_tox_thresh=0.2,
    tan_to_train_set=0.5,
    train_set_path="",
    train_set_smiles_col="SMILES",
    train_set_hit_col="",
    train_set_just_actives=False,
    train_set_hit_thresh=0,
    patterns=[],
    orig_mol_tan_thresh=1.0,
    orig_mol=None,
    display=True,
):
    """
    Filter CREM DataFrame.

    Involves deduplicating,
    gating on several column values,
    and similarity to predefined compound sets.

    :param allmolsdf: DataFrame containing all molecules
    :param smi_col: Name of the SMILES column
    :param hit_col: Name of the hit column
    :param out_dir: Name of the output directory
    :param hit_thresh: Hit threshold
    :param sascore_thresh: SAScore threshold
    :param tan_to_abx: Tanimoto similarity threshold to antibiotics
    :param abx_path: Path to antibiotics data
    :param abx_smiles_col: Name of the SMILES column in antibiotics data
    :param hepg2_tox_thresh: HepG2 toxicity threshold
    :param prim_tox_thresh: Primary toxicity threshold
    :param tan_to_train_set: Tanimoto similarity threshold to train set
    :param train_set_path: Path to train set data
    :param train_set_smiles_col: Name of the SMILES column in train set data
    :param train_set_hit_col: Name of the hit column in train set data
    :param train_set_just_actives: Flag for considering only actives in TS
    :param train_set_hit_thresh: Hit threshold in train set data
    :param patterns: List of patterns to exclude
    :param orig_mol_tan_thresh: Tanimoto similarity threshold to original cpd
    :param orig_mol: Original molecule
    :param display: Flag to display plots
    :return: Filtered DataFrame
    """
    # deduplicate
    print("original length: ", len(allmolsdf))
    df = allmolsdf.drop_duplicates(smi_col)
    df = df.reset_index(drop=True)
    print("deduplicated on smiles: ", len(df))

    # gate on score
    if display:
        hist_plot(df, hit_col, "Chemprop Score", "Round")
    df = df[df[hit_col] > hit_thresh]
    print("scores > " + str(hit_thresh) + ": ", len(df))

    # gate on synthesizability
    mols = [Chem.MolFromSmiles(smi) for smi in list(df[smi_col])]
    df["sa_score"] = [sascorer.calculateScore(mol) for mol in mols]
    if display:
        hist_plot(df, "sa_score", "Synthetic Complexity", "Round")
    keep_indices = [
        sascore < sascore_thresh for sascore in list(
            df["sa_score"])]
    df = df[keep_indices].reset_index(drop=True)
    print("SAScore < " + str(sascore_thresh) + ": ", len(df))

    # gate on tanimoto similarity to abx
    _, abx_mols = process_molset(abx_path, abx_smiles_col)
    abx_fps = [Chem.RDKFingerprint(mol) for mol in abx_mols]
    mols = [Chem.MolFromSmiles(smi) for smi in list(df[smi_col])]
    query_fps = [Chem.RDKFingerprint(mol) for mol in mols]
    df["max_tan_sim_to_abx"] = [
        max(DataStructs.BulkTanimotoSimilarity(query_fp, abx_fps))
        for query_fp in query_fps
    ]
    if display:
        hist_plot(
            df,
            "max_tan_sim_to_abx",
            "Max Tan Sim to Known Antibiotics",
            "Round")
    keep_indices = [
        max_tan < tan_to_abx for max_tan in list(
            df["max_tan_sim_to_abx"])]
    df = df[keep_indices].reset_index(drop=True)
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    print("Tan Sim to Abx < " + str(tan_to_abx) + ": ", len(df))

    # gate on toxicity scores
    _, hepg2_toxs = calculateScoreThruToxModel(
        list(df[smi_col]), out_dir, "_temp_predictions.csv", "hepg2"
    )
    df["hepg2_pred_tox"] = hepg2_toxs
    if display:
        hist_plot(df, "hepg2_pred_tox", "HepG2 Tox Model Score", "Round")
    keep_indices = [
        tox < hepg2_tox_thresh for tox in list(
            df["hepg2_pred_tox"])]
    df = df[keep_indices].reset_index(drop=True)
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    print("HepG2 pred tox < " + str(hepg2_tox_thresh) + ": ", len(df))

    _, prim_toxs = calculateScoreThruToxModel(
        list(df[smi_col]), out_dir, "_temp_predictions.csv", "primary"
    )
    df["primary_pred_tox"] = prim_toxs
    if display:
        hist_plot(df, "primary_pred_tox", "Primary Tox Model Score", "Round")
    keep_indices = [
        tox < prim_tox_thresh for tox in list(
            df["primary_pred_tox"])]
    df = df[keep_indices].reset_index(drop=True)
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    print("Primary pred tox < " + str(prim_tox_thresh) + ": ", len(df))

    # gate on tanimoto similarity to train set
    _, ts_mols = process_molset(
        train_set_path,
        train_set_smiles_col,
        train_set_hit_col,
        train_set_just_actives,
        train_set_hit_thresh,
    )
    ts_fps = [Chem.RDKFingerprint(mol) for mol in ts_mols]
    query_fps = [Chem.RDKFingerprint(mol) for mol in mols]
    df["max_tan_sim_to_ts"] = [
        max(DataStructs.BulkTanimotoSimilarity(query_fp, ts_fps))
        for query_fp in query_fps
    ]
    if display:
        hist_plot(df, "max_tan_sim_to_ts", "Max Tan Sim to Train Set", "Round")
    keep_indices = [
        max_tan < tan_to_train_set for max_tan in list(df["max_tan_sim_to_ts"])
    ]
    df = df[keep_indices].reset_index(drop=True)
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    print("Max Tan Sim to TS < " + str(tan_to_train_set) + ": ", len(df))

    # exclude patterns
    if len(patterns) > 0:  # patterns is list of mols
        for pattern in patterns:
            exclude_indices = []
            for i, mol in enumerate(mols):
                if mol.HasSubstructMatch(pattern):
                    exclude_indices.append(i)

            df = df.iloc[[i not in exclude_indices for i in range(len(df))], :]
            mols = [m for i, m in enumerate(mols) if i not in exclude_indices]
            print("excluding pattern: ", len(df))

    # calculate tanimoto similarity to the original molecule as well
    orig_mol_fp = Chem.RDKFingerprint(orig_mol)
    query_fps = [Chem.RDKFingerprint(mol) for mol in mols]
    df["tan_sim_to_orig_mol"] = [
        DataStructs.TanimotoSimilarity(q_fp, orig_mol_fp) for q_fp in query_fps
    ]
    if display:
        hist_plot(
            df,
            "tan_sim_to_orig_mol",
            "Tanimoto Similarity to Orig Mol",
            "Round")
    keep_indices = [
        tan < orig_mol_tan_thresh for tan in list(df["tan_sim_to_orig_mol"])
    ]
    df = df[keep_indices].reset_index(drop=True)
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    print("Tan sim to orig mol < " + str(orig_mol_tan_thresh) + ": ", len(df))

    return df


def analyze_crem_df(df, num_rounds=5):
    """
    Analyzes the composition of the DataFrame.

    This function takes a DataFrame containing molecular data as input
    and prints the counts of molecules from each method and round.

    :param df: DataFrame containing molecular data
    :param num_rounds: number of rounds we did optimization over
    """
    print("grow", len(df[df["Grow_or_Mut"] == "grow"]))
    print("mutate", len(df[df["Grow_or_Mut"] == "mutate"]))
    print("regular score", len(df[df["Score"] == "chemprop_score"]))
    print("modified score", len(df[df["Score"] == "modified_score"]))
    for i in range(num_rounds):
        print("round " + str(i), len(df[df["Round"] == "Round " + str(i)]))
