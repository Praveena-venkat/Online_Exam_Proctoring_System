"""
AI Proctoring System — main.py
Team: Null n Void | Guide: Saurav Ghosh | Dept: CSE

Features:
  1. Multiple face detection  → warning if more than 1 face visible
  2. Eye gaze detection       → warning if eyes leave the screen area
  3. Tab-switch detection     → warning if the browser/app loses focus
  Max 3 warnings → test terminated on 4th violation.

Run:
    python main.py

Requirements:
    pip install opencv-python numpy
    # Haar cascade XMLs ship with opencv-python — no extra download needed.
"""

import cv2
import time
import threading
import sys
import os

# ── Try to import the optional pynput listener for tab-switch detection ──────
try:
    from pynput import keyboard as pynput_keyboard   # noqa: F401 – just a check
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MAX_WARNINGS          = 3          # test terminates after this many warnings
WARNING_COOLDOWN      = 200.0      # seconds between repeated warnings for same issue type
WARNING_DISPLAY_TIME  = 60.0       # seconds each warning banner stays on screen
FRAME_WIDTH           = 640
FRAME_HEIGHT          = 480

# Gaze: fraction of frame width considered "in-bounds"
# Eyes must be within this horizontal band to be considered "on screen"
GAZE_LEFT_BOUND  = 0.15       # 15 % from left edge
GAZE_RIGHT_BOUND = 0.85       # 85 % from left edge  (i.e. 15 % from right)

# ─────────────────────────────────────────────────────────────────────────────
# HAAR CASCADE PATHS  (bundled with opencv-python)
# ─────────────────────────────────────────────────────────────────────────────
CASCADE_BASE = cv2.data.haarcascades

FACE_CASCADE_PATH = os.path.join(CASCADE_BASE, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH  = os.path.join(CASCADE_BASE, "haarcascade_eye.xml")

# ─────────────────────────────────────────────────────────────────────────────
# WARNING TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class WarningTracker:
    def __init__(self, max_warnings: int):
        self.max_warnings    = max_warnings
        self.total_warnings  = 0
        self.last_warned_at  = {}   # key → timestamp
        self.lock            = threading.Lock()
        self.terminated      = False

        # On-screen banner state for the most recent warning
        self.active_warning  = None   # dict: {"text": str, "started_at": float}
        # Once the warning that hits max_warnings is issued, this holds the
        # timestamp at which the test should actually be terminated
        # (i.e. issue_time + WARNING_DISPLAY_TIME). None = no pending termination.
        self.terminate_at    = None

    def issue(self, key: str, message: str, cooldown: float = WARNING_COOLDOWN) -> bool:
        """
        Issue a warning for `key`.  Returns True if a new warning was logged,
        False if it was suppressed by cooldown or test already terminated.

        Every logged warning is shown on screen for WARNING_DISPLAY_TIME
        seconds. Termination is NOT immediate on the final warning — it is
        scheduled to happen only after that final warning has finished its
        on-screen display window (handled by check_and_terminate()).
        """
        with self.lock:
            if self.terminated:
                return False

            now = time.time()
            last = self.last_warned_at.get(key, 0)
            if now - last < cooldown:
                return False            # still in cooldown — suppress

            self.last_warned_at[key] = now
            self.total_warnings += 1
            remaining = self.max_warnings - self.total_warnings

            border = "=" * 60
            print(f"\n{border}")
            print(f"  ⚠️  WARNING {self.total_warnings}/{self.max_warnings}: {message}")
            if remaining > 0:
                print(f"  You have {remaining} warning(s) remaining.")
            print(f"{border}\n")

            # Set/replace the on-screen banner — visible for WARNING_DISPLAY_TIME seconds
            self.active_warning = {
                "text": f"WARNING {self.total_warnings}/{self.max_warnings}: {message}",
                "started_at": now,
            }

            if self.total_warnings >= self.max_warnings:
                self.terminate_at = now + WARNING_DISPLAY_TIME
                print(f"  ⏳ Final warning. Test will terminate in {int(WARNING_DISPLAY_TIME)}s "
                      f"(after this warning has been on screen).\n")

            return True

    def get_active_warning(self):
        """
        Returns (text, seconds_left) for the warning banner that should
        currently be displayed on screen, or None if nothing should be shown.
        """
        with self.lock:
            if not self.active_warning:
                return None
            elapsed   = time.time() - self.active_warning["started_at"]
            remaining = WARNING_DISPLAY_TIME - elapsed
            if remaining <= 0:
                return None
            return self.active_warning["text"], remaining

    def check_and_terminate(self) -> bool:
        """
        Call this once per frame. Finalizes termination only once the final
        warning's on-screen display window has fully elapsed. Returns True
        the moment termination becomes final (and on every call after).
        """
        with self.lock:
            if self.terminated:
                return True
            if self.terminate_at is not None and time.time() >= self.terminate_at:
                self.terminated = True
                print("\n" + "!" * 60)
                print("  🚫  TEST TERMINATED — Maximum warnings reached.")
                print("!" * 60 + "\n")
                return True
            return False

    @property
    def is_terminated(self) -> bool:
        with self.lock:
            return self.terminated


# ─────────────────────────────────────────────────────────────────────────────
# TAB-SWITCH DETECTOR  (keyboard shortcut heuristic via pynput)
# ─────────────────────────────────────────────────────────────────────────────
class TabSwitchDetector:
    """
    Listens for common tab-switch / window-switch shortcuts:
      Alt+Tab, Ctrl+Tab, Ctrl+W, Win/Cmd key, etc.

    NOTE: A more robust approach is to use a GUI framework (tkinter / PyQt)
          and detect FocusOut events on the exam window.  The keyboard-listener
          approach works without a GUI and is suitable for a CLI proctoring tool.
    """

    SWITCH_COMBOS = {
        frozenset(["alt", "tab"]),
        frozenset(["ctrl", "tab"]),
        frozenset(["ctrl", "shift", "tab"]),
        frozenset(["ctrl", "w"]),
        frozenset(["super"]),           # Windows key
        frozenset(["cmd"]),             # macOS Cmd key
        frozenset(["alt", "f4"]),
    }

    def __init__(self, tracker: WarningTracker):
        self.tracker      = tracker
        self._held        = set()
        self._listener    = None
        self._active      = False

    def _on_press(self, key):
        try:
            name = key.char.lower() if hasattr(key, "char") and key.char else key.name.lower()
        except AttributeError:
            return
        self._held.add(name)
        self._check_combo()

    def _on_release(self, key):
        try:
            name = key.char.lower() if hasattr(key, "char") and key.char else key.name.lower()
        except AttributeError:
            return
        self._held.discard(name)

    def _check_combo(self):
        current = frozenset(self._held)
        for combo in self.SWITCH_COMBOS:
            if combo.issubset(current):
                self.tracker.issue(
                    "tab_switch",
                    "Tab / Window switch detected! Stay on the exam."
                )
                break

    def start(self):
        if not PYNPUT_OK:
            print("[TabSwitchDetector] pynput not installed — tab-switch detection disabled.")
            print("  Install with:  pip install pynput\n")
            return
        from pynput import keyboard as kb
        self._listener = kb.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
        self._active = True

    def stop(self):
        if self._listener and self._active:
            self._listener.stop()
            self._active = False


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCTORING  (face count + eye gaze)
# ─────────────────────────────────────────────────────────────────────────────
class VideoProctor:
    def __init__(self, tracker: WarningTracker):
        self.tracker        = tracker
        self.face_cascade   = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade    = cv2.CascadeClassifier(EYE_CASCADE_PATH)

        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load face cascade from {FACE_CASCADE_PATH}")
        if self.eye_cascade.empty():
            raise RuntimeError(f"Could not load eye cascade from {EYE_CASCADE_PATH}")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _detect_faces(self, gray):
        return self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

    def _detect_eyes_in_roi(self, gray_roi):
        return self.eye_cascade.detectMultiScale(
            gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

    def _gaze_out_of_bounds(self, ex, ew, frame_width, face_x):
        """
        True when the detected eye centre (in absolute frame coords) is outside
        the allowed horizontal band — suggesting the attendee is looking away.
        """
        eye_center_abs = face_x + ex + ew // 2
        left_limit  = int(frame_width * GAZE_LEFT_BOUND)
        right_limit = int(frame_width * GAZE_RIGHT_BOUND)
        return eye_center_abs < left_limit or eye_center_abs > right_limit

    # ── drawing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_overlay(frame, text, color=(0, 0, 255), y_offset=0):
        cv2.putText(
            frame, text,
            (10, 30 + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    @staticmethod
    def _draw_warning_banner(frame, text, seconds_left):
        """Persistent on-screen warning banner, visible for WARNING_DISPLAY_TIME seconds."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Disappears in {int(seconds_left)}s",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    @staticmethod
    def _draw_terminated_banner(frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "TEST TERMINATED", (max(10, w // 2 - 170), h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[VideoProctor] ERROR: Cannot open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("[VideoProctor] Webcam started.  Press 'q' to quit manually.\n")

        while True:
            # Becomes True only once the final warning has finished its
            # WARNING_DISPLAY_TIME on-screen window — this is the actual
            # termination trigger, not the moment the warning was issued.
            just_terminated = self.tracker.check_and_terminate()

            ret, frame = cap.read()
            if not ret:
                print("[VideoProctor] Failed to read frame — retrying…")
                time.sleep(0.1)
                continue

            frame_w = frame.shape[1]
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray    = cv2.equalizeHist(gray)

            faces = self._detect_faces(gray)

            # ── STATUS label ────────────────────────────────────────────────
            warnings_left = self.tracker.max_warnings - self.tracker.total_warnings
            status_text   = (
                f"Warnings: {self.tracker.total_warnings}/{self.tracker.max_warnings}"
            )
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # ── MULTIPLE FACES ───────────────────────────────────────────────
            num_faces = len(faces)
            if num_faces > 1:
                self.tracker.issue(
                    "multiple_faces",
                    f"Multiple people detected ({num_faces} faces)! Only the attendee should be visible."
                )
                self._draw_overlay(frame, f"⚠ MULTIPLE FACES ({num_faces})", (0, 0, 255))

            elif num_faces == 0:
                # No face at all — could be attendee left or camera issue
                self._draw_overlay(frame, "⚠ NO FACE DETECTED", (0, 140, 255))
            else:
                self._draw_overlay(frame, "✓ Face OK", (0, 200, 0))

            # ── EYE GAZE ────────────────────────────────────────────────────
            gaze_warning = False
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)

                # only analyse the upper 60 % of the face (eye region)
                roi_gray  = gray[fy: fy + int(fh * 0.6), fx: fx + fw]
                eyes      = self._detect_eyes_in_roi(roi_gray)

                if len(eyes) == 0 and num_faces == 1:
                    # Eyes not found — attendee might be looking far away or down
                    gaze_warning = True
                    self.tracker.issue(
                        "gaze_missing",
                        "Eye gaze not detected! Keep your eyes on the screen."
                    )
                else:
                    for (ex, ey, ew, eh) in eyes:
                        # absolute coords for drawing
                        abs_ex = fx + ex
                        abs_ey = fy + ey
                        cv2.rectangle(
                            frame,
                            (abs_ex, abs_ey),
                            (abs_ex + ew, abs_ey + eh),
                            (0, 255, 255), 1
                        )
                        if self._gaze_out_of_bounds(ex, ew, frame_w, fx):
                            gaze_warning = True

                if gaze_warning:
                    self.tracker.issue(
                        "gaze_out",
                        "Eye gaze moved outside screen bounds! Focus on the exam."
                    )
                    self._draw_overlay(frame, "⚠ GAZE OUT OF BOUNDS", (0, 0, 255), y_offset=35)
                elif num_faces == 1:
                    self._draw_overlay(frame, "✓ Gaze OK", (0, 200, 0), y_offset=35)

            # ── PERSISTENT WARNING BANNER (visible for WARNING_DISPLAY_TIME secs) ──
            active = self.tracker.get_active_warning()
            if active:
                banner_text, seconds_left = active
                self._draw_warning_banner(frame, banner_text, seconds_left)

            if just_terminated:
                self._draw_terminated_banner(frame)

            cv2.imshow("AI Proctoring System — Null n Void", frame)
            key = cv2.waitKey(1) & 0xFF

            if just_terminated:
                # Hold the "TEST TERMINATED" frame on screen briefly before closing.
                time.sleep(2)
                break

            if key == ord("q"):
                print("[VideoProctor] Manual quit.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[VideoProctor] Camera released.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   AI PROCTORING SYSTEM — Null n Void")
    print("   Guide: Saurav Ghosh  |  Dept: CSE")
    print("=" * 60)
    print(f"  Max warnings before termination : {MAX_WARNINGS}")
    print(f"  Warning on-screen duration       : {WARNING_DISPLAY_TIME:.0f}s")
    print(f"  Warning cooldown                : {WARNING_COOLDOWN}s")
    print(f"  Gaze horizontal bounds          : {int(GAZE_LEFT_BOUND*100)}% – {int(GAZE_RIGHT_BOUND*100)}%")
    print("=" * 60 + "\n")

    tracker     = WarningTracker(MAX_WARNINGS)
    tab_detect  = TabSwitchDetector(tracker)
    video       = VideoProctor(tracker)

    # Start keyboard listener in background thread
    tab_detect.start()

    try:
        video.run()                 # blocks until terminated or user quits
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        tab_detect.stop()

    if tracker.is_terminated:
        print("\n🚫  Exam session TERMINATED due to repeated violations.")
        sys.exit(1)
    else:
        print("\n✅  Exam session ended normally.")
        sys.exit(0)


if __name__ == "__main__":
    main()
