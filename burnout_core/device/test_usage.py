import time
from burnout_core.device.usage_tracker import (
    get_active_window,
    get_idle_time,
    collect_usage
)

print("Active application:", get_active_window())
print("Don't touch the keyboard or mouse for 5 seconds...")
time.sleep(5)
print("Idle time:", get_idle_time(), "seconds")

print("Use your computer normally for 20 seconds...")
print(collect_usage(duration=20, interval=5, idle_limit=3))
