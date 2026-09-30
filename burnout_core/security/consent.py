import json
import tkinter as tk
from pathlib import Path

CONSENT_PATH = Path.home() / "burnout_usage" / "consent.json"


def has_remembered_choice():
    """Check if the user previously chose to remember their answer."""

    if not CONSENT_PATH.exists():
        return None

    data = json.loads(CONSENT_PATH.read_text())

    if not data.get("remember"):
        return None

    return data.get("accepted")


def save_choice(accepted, remember):
    """Save the user's choice, if they asked to remember it."""

    CONSENT_PATH.parent.mkdir(exist_ok=True)

    CONSENT_PATH.write_text(json.dumps({
        "accepted": accepted,
        "remember": remember
    }))


def show_consent_screen():
    """Show the consent popup, unless a remembered choice exists."""

    remembered = has_remembered_choice()

    if remembered is not None:
        return remembered

    result = {"accepted": False}
    root = tk.Tk()
    root.title("Privacy & Consent")
    root.configure(bg="#f5f5f5")

    window_width = 700
    window_height = 660

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    
    remember_var = tk.BooleanVar(value=False)

    title = tk.Label(
        root,
        text="🔒 Privacy & Data Consent",
        font=("Segoe UI", 22, "bold"),
        bg="#f5f5f5"
    )
    title.pack(pady=(30, 15))

    text = """This application processes sensitive usage and
burnout-related information.

Data collected:
  •  Application usage patterns
  •  Screen / idle time
  •  Burnout prediction results
  •  Wellness information

Why:
  To calculate burnout risk and provide timely interventions.

Storage:
  Your data is stored locally on this device and encrypted.
  You can delete your stored data at any time.

  This application does not sell or share your data with
  advertising or analytics services."""

    label = tk.Label(
        root,
        text=text,
        font=("Segoe UI", 12),
        bg="#f5f5f5",
        justify="left",
        wraplength=600
    )
    label.pack(pady=10, padx=40)

    remember_check = tk.Checkbutton(
        root,
        text="Remember my choice",
        variable=remember_var,
        font=("Segoe UI", 11),
        bg="#f5f5f5",
        activebackground="#f5f5f5",
        cursor="hand2"
    )
    remember_check.pack(pady=(10, 0))

    def accept():
        result["accepted"] = True
        save_choice(True, remember_var.get())
        root.destroy()

    def decline():
        result["accepted"] = False
        save_choice(False, remember_var.get())
        root.destroy()

    button_frame = tk.Frame(root, bg="#f5f5f5")
    button_frame.pack(pady=20)

    agree_btn = tk.Button(
        button_frame,
        text="I Agree",
        command=accept,
        width=16,
        font=("Segoe UI", 12, "bold"),
        bg="#2e7d32",
        fg="white",
        activebackground="#1b5e20",
        activeforeground="white",
        relief="flat",
        cursor="hand2"
    )
    agree_btn.grid(row=0, column=0, padx=10)

    decline_btn = tk.Button(
        button_frame,
        text="Decline",
        command=decline,
        width=16,
        font=("Segoe UI", 12),
        bg="#e0e0e0",
        fg="#333333",
        activebackground="#bdbdbd",
        relief="flat",
        cursor="hand2"
    )
    decline_btn.grid(row=0, column=1, padx=10)

    root.mainloop()

    return result["accepted"]


def delete_data_button(parent, delete_function):
    button = tk.Button(
        parent,
        text="Delete All My Data",
        command=delete_function,
        width=20
    )

    button.pack(pady=15)
