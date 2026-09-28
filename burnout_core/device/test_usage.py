from burnout_core.device.usage_tracker import (
    collect_usage, save_daily_usage, summarize_usage
)

print("Use your computer normally for 30 seconds...")
usage = collect_usage(duration=30, interval=5)
print(usage)

totals = save_daily_usage(usage)
print(summarize_usage(totals))
