import time
import psutil
import win32api
import win32gui
import win32process


def get_active_window():
    """Return the name of the currently active application."""

    try:
        hwnd = win32gui.GetForegroundWindow()

        if not hwnd:
            return "Unknown"

        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        process = psutil.Process(pid)

        return process.name()

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "Unknown"


def get_idle_time():
    """Return computer idle time in seconds."""

    last_input = win32api.GetLastInputInfo()
    current_tick = win32api.GetTickCount()

    return (current_tick - last_input) / 1000


def collect_usage(duration=60, interval=5, idle_limit=60):
    """
    Track active applications for a period of time.
    Time is not counted while the computer has been idle
    for longer than idle_limit seconds.

    Returns:
        dictionary containing usage statistics.
    """

    usage = {}
    start_time = time.time()

    while time.time() - start_time < duration:

        app = get_active_window()

        if app not in usage:
            usage[app] = 0

        if get_idle_time() < idle_limit:
            usage[app] += interval

        time.sleep(interval)

    return usage
