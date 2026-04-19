import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_timeline_graph(events, exam_duration=600, filename="timeline.png"):
    import matplotlib.pyplot as plt
    import os

    REPORTS_DIR = "reports"
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not events:
        return None

    # 🔥 convert events to seconds remaining
    event_map = {}

    for ev in events:
        try:
            minutes, seconds = map(int, ev["time"].split(":"))
            remaining = minutes * 60 + seconds

            if "Phone" in ev["type"] or "Face" in ev["type"]:
                value = 2
            elif "Gaze" in ev["type"] or "Lip" in ev["type"]:
                value = 1
            else:
                value = 0

            event_map[remaining] = value

        except:
            continue

    # 🔥 build FULL timeline (0 → exam_duration)
    values = []
    labels = []

    for t in range(exam_duration, -1, -1):  # countdown

        value = event_map.get(t, 0)  # default = normal

        values.append(value)

        minutes = t // 60
        seconds = t % 60
        labels.append(f"{minutes:02d}:{seconds:02d}")

    x = list(range(len(values)))

    plt.figure(figsize=(60, 3))

    plt.plot(x, values, color='red', linewidth=2)
    plt.scatter(x, values, color='red', s=10)

    # 🔥 FIX GAP
    plt.margins(x=0)
    plt.xlim(0, len(values)-1)

    plt.title("Behavior Timeline")
    plt.xlabel("Time Remaining")
    plt.ylabel("Activity")

    plt.ylim(0, 2.5)

    step = max(1, len(labels)//10)
    plt.xticks(x[::step], labels[::step], rotation=30)

    plt.yticks([0, 1, 2], ["Normal", "Warning", "Suspicious"])

    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path)
    plt.close()

    return filename