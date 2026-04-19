import numpy as np
import math

def compute_metrics(gaze_points, events, timeline_data, screen_w=1920, screen_h=1080):

    if not gaze_points:
        return {}

    xs = np.array([p[0] for p in gaze_points])
    ys = np.array([p[1] for p in gaze_points])

    # -------------------------------
    # 1. GAZE STABILITY
    # -------------------------------
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)

    std_x = np.std(xs)
    std_y = np.std(ys)

    # -------------------------------
    # 2. MAE (from screen center)
    # -------------------------------
    center_x = screen_w / 2
    center_y = screen_h / 2

    mae = np.mean(np.abs(xs - center_x) + np.abs(ys - center_y))

    # -------------------------------
    # 3. MSE
    # -------------------------------
    mse = np.mean((xs - center_x)**2 + (ys - center_y)**2)

    # -------------------------------
    # 4. MEAN ANGULAR ERROR
    # -------------------------------
    angles = []
    for x, y in zip(xs, ys):
        dx = x - center_x
        dy = y - center_y
        angle = math.degrees(math.atan2(dy, dx))
        angles.append(abs(angle))

    mange = np.mean(angles)

    # -------------------------------
    # 5. ATTENTION SCORE
    # -------------------------------
    # 🔥 better attention region (50% screen)
    x_min = screen_w * 0.25
    x_max = screen_w * 0.75
    y_min = screen_h * 0.25
    y_max = screen_h * 0.75

    attentive = np.sum(
        (xs >= x_min) & (xs <= x_max) &
        (ys >= y_min) & (ys <= y_max)
    )

    total = len(xs)
    attention_score = (attentive / total) * 100

    # -------------------------------
    # 6. DISTRACTION RATIO
    # -------------------------------
    distraction_ratio = 100 - attention_score

    # -------------------------------
    # 7. EVENT METRICS
    # -------------------------------
    total_events = len(events)

    serious_events = sum(
        1 for e in events
        if "Phone" in e["type"] or "Face" in e["type"]
    )

    warning_events = sum(
        1 for e in events
        if "Gaze" in e["type"] or "Lip" in e["type"]
    )

    # simple proxy accuracy
    if total_events == 0:
        event_accuracy = 100
    else:
        event_accuracy = (warning_events / total_events) * 100

    # -------------------------------
    # 8. ATTENTION (EVENT BASED)
    # -------------------------------

    timeline = timeline_data if timeline_data else []

    total = len(timeline) if timeline else 1

    normal_count = sum(1 for t in timeline if t["event"] == "normal")

    attention_score = (normal_count / total) * 100
    distraction_ratio = 100 - attention_score
    total_events = len(events) if events else 1



    # -------------------------------
    # 9. TRUST SCORE
    # -------------------------------

    trust_score = 100 \
        - (serious_events * 5) \
        - (warning_events * 2)

    trust_score = max(0, min(100, trust_score))

    

    # -------------------------------
    return {
        "mean_x": round(mean_x, 2),
        "mean_y": round(mean_y, 2),
        "std_x": round(std_x, 2),
        "std_y": round(std_y, 2),
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "mange": round(mange, 2),
        "attention_score": round(attention_score, 2),
        "distraction_ratio": round(distraction_ratio, 2),
        "trust_score": round(trust_score, 2),
        "event_accuracy": round(event_accuracy, 2),
        "total_events": total_events,
        "serious_events": serious_events,
        "warning_events": warning_events
    }