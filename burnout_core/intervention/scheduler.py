import json
from pathlib import Path
from datetime import datetime


CONFIG_PATH = Path("config/intervention.json")


def load_schedule():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


def is_focus_window(schedule):
    now = datetime.now().time()

    start = datetime.strptime(
        schedule["start_time"], "%H:%M"
    ).time()

    end = datetime.strptime(
        schedule["end_time"], "%H:%M"
    ).time()

    return start <= now <= end
