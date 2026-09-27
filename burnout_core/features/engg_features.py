def engineer_features(data):
    data = data.copy()

    data["stress_sleep_ratio"] = (
        data["stress_level"] /
        (data["sleep_hours"] + 1)
    )

    data["mental_pressure"] = (
        data["anxiety_score"] +
        data["depression_score"] +
        data["exam_pressure"]
    )

    data["wellness_score"] = (
        data["physical_activity"] +
        data["social_support"] -
        data["stress_level"]
    )

    data["digital_overload"] = (
        data["screen_time"] *
        data["internet_usage"]
    )

    data["sleep_quality"] = (
        data["sleep_hours"] /
        (data["screen_time"] + 1)
    )

    data["stress_index"] = (
        data["stress_level"] *
        data["exam_pressure"]
    )

    data["lifestyle_balance"] = (
        data["physical_activity"] +
        data["social_support"] -
        data["screen_time"]
    )

    return data
