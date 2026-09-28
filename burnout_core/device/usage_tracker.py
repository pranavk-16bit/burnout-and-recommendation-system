import ipaddress
import json
import time
from datetime import date
from pathlib import Path

import psutil
import win32api
import win32gui
import win32process

LOG_DIR = Path.home() / "burnout_usage"
LOG_DIR.mkdir(exist_ok=True)


def get_active_window_info():
    """Return (app_name, pid) of the active window."""

    try:
        hwnd = win32gui.GetForegroundWindow()

        if not hwnd:
            return "Unknown", None

        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        return psutil.Process(pid).name(), pid

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "Unknown", None


def get_active_window():
    """Return the name of the currently active application."""

    return get_active_window_info()[0]


def get_idle_time():
    """Return computer idle time in seconds."""

    last_input = win32api.GetLastInputInfo()
    current_tick = win32api.GetTickCount()

    return (current_tick - last_input) / 1000


def is_using_internet(pid):
    """True if the app (or its helper processes) has an
    established connection to a non-local address."""

    if pid is None:
        return False

    try:
        proc = psutil.Process(pid)

        for p in [proc] + proc.children(recursive=True):
            for conn in p.net_connections(kind="inet"):

                if conn.status != psutil.CONN_ESTABLISHED or not conn.raddr:
                    continue

                ip = ipaddress.ip_address(conn.raddr.ip)

                if not (ip.is_private or ip.is_loopback):
                    return True

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return False


def collect_usage(duration=60, interval=5, idle_limit=60,
                  min_bytes=250_000):
    """
    Track screen and internet seconds per app.
    Idle time is not counted. An interval counts as internet
    use only if the active app has a connection AND at least
    min_bytes moved over the network during the interval.
    """

    usage = {}
    start_time = time.time()

    while time.time() - start_time < duration:

        app, pid = get_active_window_info()

        if app not in usage:
            usage[app] = {"screen": 0, "internet": 0}

        connected = is_using_internet(pid)
        before = get_net_bytes()

        time.sleep(interval)

        moved = get_net_bytes() - before

        if get_idle_time() < idle_limit:
            usage[app]["screen"] += interval

            if connected and moved >= min_bytes:
                usage[app]["internet"] += interval

    return usage




def save_daily_usage(usage):
    """Add this session's totals to today's log file."""

    path = LOG_DIR / f"{date.today().isoformat()}.json"

    saved = json.loads(path.read_text()) if path.exists() else {}

    for app, t in usage.items():
        entry = saved.setdefault(app, {"screen": 0, "internet": 0})
        entry["screen"] += t["screen"]
        entry["internet"] += t["internet"]

    path.write_text(json.dumps(saved, indent=2))

    return saved


def summarize_usage(usage):
    """Convert seconds-per-app into hours for the model."""

    screen = sum(t["screen"] for t in usage.values())
    internet = sum(t["internet"] for t in usage.values())

    return {
        "screen_time": screen / 3600,
        "internet_usage": internet / 3600
    }

def get_net_bytes():
    """Total bytes sent + received by this computer so far."""

    c = psutil.net_io_counters()
    return c.bytes_sent + c.bytes_recv
