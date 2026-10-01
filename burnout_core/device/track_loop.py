import time
from burnout_core.device.usage_tracker import collect_usage, save_daily_usage

BURST_DURATION = 30
INTERVAL = 5
SLEEP_BETWEEN = 15 * 60


def main():
    print("Usage tracker started. Press Ctrl+C to stop.")

    try:
        while True:
            usage = collect_usage(duration=BURST_DURATION, interval=INTERVAL)
            save_daily_usage(usage)

            time.sleep(SLEEP_BETWEEN)

    except KeyboardInterrupt:
        print("\nUsage tracker stopped.")


if __name__ == "__main__":
    main()
