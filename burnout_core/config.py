import os
from pathlib import Path

from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = os.getenv("BURNOUT_DATA_PATH")

if not DATA_PATH:
    raise RuntimeError(
        "BURNOUT_DATA_PATH is not set in the .env file."
    )

DATA_PATH = Path(DATA_PATH)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
VISUAL_DIR = BASE_DIR / "visuals"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "burnout_model.pkl"
