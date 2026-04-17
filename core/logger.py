import cv2
import os
import json
from datetime import datetime

# ── Snapshots directory ────────────────────────────────────────────────────────
SNAPSHOTS_DIR = "static/snapshots"
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


class ExamLogger:
    """
    Tracks all proctoring events during an exam session.
    Saves snapshots and builds behavioral timeline.
    """

    def __init__(self, student_id, exam_id):
        self.student_id   = student_id
        self.exam_id      = exam_id
        self.session_id   = f"{student_id}_{exam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.events       = []          # list of event dicts
        self.gaze_points  = []          # list of (x, y) for heatmap
        self.snapshots    = []          # list of snapshot info dicts

        # Counters
        self.phone_count     = 0
        self.face_count      = 0
        self.lip_count       = 0
        self.gaze_away_count = 0

        # Cooldown tracking (avoid logging same event 100 times)
        self.last_event_time  = {}
        self.cooldown_seconds = 3       # min seconds between same event type

        print(f"Logger started: {self.session_id}")

    def _get_timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def _cooldown_ok(self, event_type):
        """
        Returns True if enough time has passed since
        last event of this type (avoids spam logging)
        """
        now = datetime.now()
        last = self.last_event_time.get(event_type)
        if last is None:
            self.last_event_time[event_type] = now
            return True
        diff = (now - last).total_seconds()
        if diff >= self.cooldown_seconds:
            self.last_event_time[event_type] = now
            return True
        return False

    def log_event(self, event_type, frame=None, save_snapshot=True):
        """
        Logs a proctoring event.
        event_type: str like 'Phone Detected', 'Multiple Faces', etc.
        frame     : webcam frame (numpy array) for snapshot
        """
        if not self._cooldown_ok(event_type):
            return  # skip — too soon since last same event

        timestamp = self._get_timestamp()

        # Update counters
        if "Phone"    in event_type: self.phone_count     += 1
        if "Face"     in event_type: self.face_count      += 1
        if "Lip"      in event_type: self.lip_count       += 1
        if "Gaze"     in event_type: self.gaze_away_count += 1

        # Save snapshot if frame provided
        snap_filename = None
        if save_snapshot and frame is not None:
            snap_filename = self._save_snapshot(frame, event_type, timestamp)

        # Add to events list
        event = {
            "time":     timestamp,
            "type":     event_type,
            "snapshot": snap_filename
        }
        self.events.append(event)

        if snap_filename:
            self.snapshots.append({
                "file":  snap_filename,
                "event": event_type,
                "time":  timestamp
            })

        print(f"[{timestamp}] EVENT: {event_type}"
              + (f" → {snap_filename}" if snap_filename else ""))

    def _save_snapshot(self, frame, event_type, timestamp):
        """
        Saves a webcam frame as snapshot image.
        Returns filename.
        """
        # Clean event type for filename
        clean_type = event_type.replace(" ", "_").replace("!", "")
        filename   = f"{self.session_id}_{clean_type}_{timestamp.replace(':','')}.jpg"
        filepath   = os.path.join(SNAPSHOTS_DIR, filename)

        # Add red border + label to snapshot
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Red border
        cv2.rectangle(annotated, (0,0), (w-1,h-1), (0,0,255), 8)

        # Label background
        cv2.rectangle(annotated, (0,0), (w, 40), (0,0,200), -1)

        # Label text
        cv2.putText(annotated,
                    f"{event_type} | {timestamp}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,255), 2)

        cv2.imwrite(filepath, annotated)
        return filename

    def log_gaze_point(self, x, y):
        """
        Records a gaze coordinate for heatmap generation.
        Called every time gaze is estimated.
        """
        self.gaze_points.append((x, y))

    def get_summary(self):
        """
        Returns complete session summary for report page.
        """
        return {
            "session_id":      self.session_id,
            "events":          self.events,
            "gaze_points":     self.gaze_points,
            "snapshots":       self.snapshots,
            "phone_count":     self.phone_count,
            "face_count":      self.face_count,
            "lip_count":       self.lip_count,
            "gaze_away_count": self.gaze_away_count,
            "total_events":    len(self.events)
        }

    def reset(self):
        """Clears all data — call when new exam starts"""
        self.__init__(self.student_id, self.exam_id)


# ── Global logger instance ─────────────────────────────────────────────────────
# This gets reused across Flask requests
_active_logger = None

def get_logger():
    return _active_logger

def start_logger(student_id, exam_id):
    global _active_logger
    _active_logger = ExamLogger(student_id, exam_id)
    return _active_logger

def stop_logger():
    global _active_logger
    summary = None
    if _active_logger:
        summary = _active_logger.get_summary()
    _active_logger = None
    return summary


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing logger...")

    # Create logger
    logger = ExamLogger("CB001", "EXAM001")

    # Simulate some events
    logger.log_event("Phone Detected",    frame=None, save_snapshot=False)
    logger.log_event("Multiple Faces",    frame=None, save_snapshot=False)
    logger.log_event("Lip Movement",      frame=None, save_snapshot=False)
    logger.log_event("Gaze Away - Left",  frame=None, save_snapshot=False)
    logger.log_event("Gaze Away - Right", frame=None, save_snapshot=False)

    # Simulate gaze points
    import numpy as np
    for _ in range(50):
        x = int(np.random.normal(960, 200))
        y = int(np.random.normal(400, 150))
        logger.log_gaze_point(x, y)

    # Get summary
    summary = logger.get_summary()
    print(f"\nSession: {summary['session_id']}")
    print(f"Events : {summary['total_events']}")
    print(f"Phone  : {summary['phone_count']}")
    print(f"Faces  : {summary['face_count']}")
    print(f"Lips   : {summary['lip_count']}")
    print(f"Gaze   : {summary['gaze_away_count']}")
    print(f"Gaze pts: {len(summary['gaze_points'])}")
    print("\nTimeline:")
    for ev in summary["events"]:
        print(f"  [{ev['time']}] {ev['type']}")
        