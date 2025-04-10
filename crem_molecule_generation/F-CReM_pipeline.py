import os
from crem_pipeline import run_crem  # Imports the main function that runs the CReM algorithm

# Set a base directory for storing intermediate predictions and results.
DIR = '../out/crem/'  # Intermediate folder for predictions/results

# Define the original molecule and fragment using SMILES strings.
# These SMILES strings represent the starting chemical structures.
ORIG_MOL_SMI = 'O=C(NC1CC2CCC1O2)Nc3cccc(F)c3'
ORIG_FRAG_SMI = 'O=C(NC1CC2CCC1O2)Nc3cccc(F)c3' # F2 fragment

# Parameter settings for the molecule generation process:
# For "grow" operations, MAX_ATOM_RANGE and MIN_ATOM_RANGE determine the range of fragment sizes 
# that will be used in the replacement process (e.g. replacing hydrogen atoms).
MAX_ATOM_RANGE = [6]
MIN_ATOM_RANGE = [0]  # 0 indicates that hydrogens (or very small fragments) are replaced

# RADIUS_RANGE defines the size (or radius) of the environment around the atom to be considered for modification.
RADIUS_RANGE = [3, 2]

# These ranges are applicable for "mutate" operations to control how much the molecule changes:
MIN_INC_RANGE = [-2]  # Only used for mutation: the minimum change in number of heavy atoms
MAX_INC_RANGE = [2]   # Only used for mutation: the maximum change in number of heavy atoms

# Settings for the number of molecules to select after each iteration:
NUM_TOP_TO_GET = 20      # Number of top (best scoring) molecules to keep
NUM_RANDOM_TO_GET = 10   # Number of randomly selected molecules to retain
NUM_ITERS = 10           # Total number of iterations for the pipeline to run

# Define which compound filters to apply.
# Options include 'pains', 'brenk', 'both', or 'none'.
CPD_FILTER = 'both'

# Set the model path, which points to the directory holding the Chemprop-based scoring model.
MODEL_PATH = '../models/Staphylococcus_aureus/checkpoints/'

# HIT_COLUMN: Name of the column in prediction output that contains the activity scores.
HIT_COLUMN = 'ACTIVITY'

# Loop over different scoring method settings. Here, only 'modified_score' is used.
# The pipeline supports two types of scoring:
# 1. Regular Chemprop score (flag True)
# 2. Modified score (incorporates additional penalties/bonus for synthesizability, toxicity, etc.),
#    which is triggered when the flag is False.
for score_path in ['modified_score']:
    if score_path == 'regular_score':
        REGULAR_SCORE = True  # Use regular Chemprop scores
    else:
        REGULAR_SCORE = False  # Use the modified scoring scheme (penalizes based on additional criteria)

    # Create an output directory specific to the chosen scoring method.
    # The directory will be: DIR + score_path + '/'
    OUT_DIR = DIR + score_path + '/'
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)  # Create the directory if it doesn't exist

    # Loop over the available molecule modification methods.
    # In this example, only 'grow' is used (adding fragments to the original molecule).
    for mut_or_grow in ['grow']:
        METHOD = mut_or_grow  # Set the current method, e.g., 'grow'

        # Run the CReM pipeline using the run_crem function from crem_pipeline_0.
        # The function is called with all defined parameters:
        # - out_dir: where all results and predictions are saved
        # - orig_frag_smi and orig_mol_smi: input SMILES strings defining the fragment and whole molecule
        # - Parameter ranges for growing or mutating the molecule
        # - Iterative settings: number of iterations, scoring method, and number of candidates to select
        # - cpd_filter: type of compound filter (PAINS/Brenk) to apply during generation
        # - model_path: path to the scoring model used within Chemprop
        # - hit_column: which column to interpret as the activity score in model predictions
        run_crem(
            out_dir=OUT_DIR,
            orig_frag_smi=ORIG_FRAG_SMI,
            orig_mol_smi=ORIG_MOL_SMI,
            max_atom_range=MAX_ATOM_RANGE,
            min_atom_range=MIN_ATOM_RANGE,
            radius_range=RADIUS_RANGE,
            min_inc_range=MIN_INC_RANGE,
            max_inc_range=MAX_INC_RANGE,
            num_iters=NUM_ITERS,
            method=METHOD,
            regular_score=REGULAR_SCORE,
            num_top_to_get=NUM_TOP_TO_GET,
            num_random_to_get=NUM_RANDOM_TO_GET,
            cpd_filter=CPD_FILTER,
            model_path=MODEL_PATH,
            hit_column=HIT_COLUMN
        )
