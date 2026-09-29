def burnout_status(score):
    thresholds = [
        (1.5, "Stable"),
        (2.3, "Watchlist")
    ]

    for limit, status in thresholds:
        if score < limit:
            return status

    return "Critical"
