from datetime import datetime

from burnout_core.app_config import config


def load_schedule():
    return {
        "start_time": "09:00",
        "end_time": "22:00",
        "break_every_minutes": config["intervention"]["break_every_minutes"]
    }


def is_focus_window(schedule):
    now = datetime.now().time()

    start = datetime.strptime(
        schedule["start_time"], "%H:%M"
    ).time()

    end = datetime.strptime(
        schedule["end_time"], "%H:%M"
    ).time()

    return start <= now <= end

