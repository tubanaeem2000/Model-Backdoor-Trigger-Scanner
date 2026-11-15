"""
config.py - All project settings in ONE place.
Change a value here and every other file uses the new value.
"""
from pathlib import Path

# Project root = the folder that contains this "src" folder
ROOT_DIR = Path(__file__).resolve().parent.parent

# Folders
DATA_DIR = ROOT_DIR / "data"        # MNIST gets downloaded here
MODELS_DIR = ROOT_DIR / "models"    # trained models (.pth) are saved here
REPORTS_DIR = ROOT_DIR / "reports"  # training results / scan reports
OUTPUTS_DIR = ROOT_DIR / "outputs"  # images / plots we generate

# Dataset settings
BATCH_SIZE = 64
NUM_CLASSES = 10          # MNIST digits 0-9
IMAGE_SIZE = 28           # MNIST images are 28 x 28 pixels
IN_CHANNELS = 3           # MNIST is grayscale, but we convert it to RGB (3 channels)
                          # so that a YELLOW trigger is possible (yellow = red + green)

# ---------------- Backdoor attack settings ----------------
TARGET_CLASS = 7                  # poisoned images get this label
POISON_RATE = 0.10                # 10% of training images get the trigger
TRIGGER_SIZE = 4                  # trigger square is 4 x 4 pixels
TRIGGER_COLOR = (1.0, 1.0, 0.0)   # yellow in RGB (values 0.0 - 1.0)
TRIGGER_MARGIN = 1                # gap from the bottom-right corner

# ---------------- Training settings ----------------
EPOCHS = 3                # full passes over the training data
LEARNING_RATE = 0.001
SEED = 42                 # fixed random seed -> same results every run

# Model file names
CLEAN_MODEL_PATH = MODELS_DIR / "clean_model.pth"
BACKDOOR_MODEL_PATH = MODELS_DIR / "backdoored_model.pth"
RESULTS_PATH = REPORTS_DIR / "training_results.json"

# ---------------- Scanner settings ----------------
# NOTE: the scanner does NOT use TARGET_CLASS or TRIGGER_* above.
# It searches blindly over these candidate patches and lets the model's behaviour reveal the backdoor.
SCAN_NUM_IMAGES = 500     # clean test images used for probing
SCAN_COLORS = {           # candidate patch colours (RGB, 0-1)
    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0), "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0),
    "white": (1.0, 1.0, 1.0), "gray": (0.5, 0.5, 0.5),
}
SCAN_SIZES = [2, 3, 4, 5, 6]                       # patch sizes in pixels
SCAN_POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
SCAN_MARGIN = 1                                    # gap from the image edge for corner patches
ASR_THRESHOLD = 0.80      # >= 80% of images pushed into ONE class  -> suspicious
FLIP_THRESHOLD = 0.50     # >= 50% of predictions changed           -> suspicious
SCAN_RESULTS_PREFIX = "scan_results"               # reports/scan_results_<model>.csv
SCAN_SUMMARY_PATH = REPORTS_DIR / "scan_summary.csv"

# Make sure the folders exist
for folder in (DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
