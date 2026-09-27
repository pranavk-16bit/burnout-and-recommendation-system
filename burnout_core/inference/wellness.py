def compute_wellness(student):

    sleep_component = (
        min(student["sleep_hours"], 9) / 9
    )

    activity_component = (
        min(student["physical_activity"], 5) / 5
    )

    support_component = (
        student["social_support"] / 10
    )

    screen_penalty = (
        min(student["screen_time"], 12) / 12
    )

    wellness = (
        sleep_component * 4
        + activity_component * 3
        + support_component * 3
        - screen_penalty * 3
    )

    return max(
        0,
        min(wellness, 10)
    )
