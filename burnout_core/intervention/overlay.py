import tkinter as tk


def show_intervention(
    message,
    level="Stable",
    mandatory_seconds=0
):
    root = tk.Tk()

    root.title("Burnout Prevention")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)

    frame = tk.Frame(root)
    frame.pack(expand=True)

    title = tk.Label(
        frame,
        text="🧘 BREAK TIME",
        font=("Arial", 40, "bold")
    )
    title.pack(pady=30)

    message_label = tk.Label(
        frame,
        text=message,
        font=("Arial", 22),
        wraplength=900
    )
    message_label.pack(pady=20)

    icon = tk.Label(
        frame,
        text="🚶 💧",
        font=("Arial", 70)
    )
    icon.pack(pady=20)

    countdown = tk.Label(
        frame,
        text="",
        font=("Arial", 18)
    )
    countdown.pack(pady=10)

    def done():
        root.destroy()

    def snooze():
        root.destroy()

    done_button = tk.Button(
        frame,
        text="Done",
        command=done,
        font=("Arial", 18),
        width=12
    )
    done_button.pack(pady=10)

    snooze_button = tk.Button(
        frame,
        text="Snooze 5 min",
        command=snooze,
        font=("Arial", 18),
        width=12
    )
    snooze_button.pack(pady=10)

    if mandatory_seconds > 0:
        snooze_button.config(state="disabled")
        done_button.config(state="disabled")

        def unlock(seconds):
            if seconds > 0:
                countdown.config(
                    text=f"Please take a break: {seconds}s"
                )
                root.after(
                    1000,
                    lambda: unlock(seconds - 1)
                )
            else:
                countdown.config(text="")
                done_button.config(state="normal")
                snooze_button.config(state="normal")

        unlock(mandatory_seconds)

    root.mainloop()
