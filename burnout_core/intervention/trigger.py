import time
from datetime import datetime
from burnout_core.app_config import config
from .scheduler import load_schedule, is_focus_window
from .overlay import show_intervention


class InterventionManager:

    def __init__(self):
        self.schedule = load_schedule()
        self.last_break = datetime.now()

    def minutes_since_break(self):
        elapsed = datetime.now() - self.last_break
        return elapsed.total_seconds() / 60

    def should_trigger(self, risk_level):

        if not is_focus_window(self.schedule):
            return False

        elapsed = self.minutes_since_break()

        return (
            elapsed >=
            self.schedule["break_every_minutes"]
        )

    def trigger(self, risk_level):

        if risk_level == "Critical":

            show_intervention(
                "Your burnout risk is critical. "
                "Please take a proper break.",
                level="Critical",
                mandatory_seconds=config["intervention"]["critical_mandatory_seconds"]
            )

        elif risk_level == "Watchlist":

            show_intervention(
                "Your burnout risk is increasing. "
                "Take a short break and reset.",
                level="Watchlist",
                mandatory_seconds=config["intervention"]["watchlist_mandatory_seconds"]
            )

        else:

            show_intervention(
                "You've been at your screen for "
                "50 minutes — time to stretch.",
                level="Stable"
            )

        self.last_break = datetime.now()

    def run(self, get_risk):

        while True:

            risk_level = get_risk()

            if self.should_trigger(risk_level):
                self.trigger(risk_level)

            time.sleep(30)
