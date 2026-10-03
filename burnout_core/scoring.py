from burnout_core.app_config import config


def burnout_status(score):
    thresholds = [
        (config["scoring"]["stable_threshold"], "Stable"),
        (config["scoring"]["watchlist_threshold"], "Watchlist")
    ]

    for limit, status in thresholds:
        if score < limit:
            return status

    return "Critical"
