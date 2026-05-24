import threading
import queue
import time
import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoachMessage:
    text:     str
    priority: int     # 1=low 2=medium 3=urgent
    category: str     # form / fatigue / injury / pace


class VoiceCoach:
    COOLDOWNS = {
        'form':    8.0,
        'fatigue': 10.0,
        'injury':  6.0,
        'pace':    12.0,
    }

    def __init__(self, enabled: bool = True,
                 audio_dir: str = "data/audio"):
        self.enabled      = enabled
        self.audio_dir    = audio_dir
        self.msg_queue    = queue.PriorityQueue()
        self.last_spoken: dict = {}
        self.speaking     = False
        self.engine       = None
        self._use_pyttsx3 = False

        os.makedirs(audio_dir, exist_ok=True)

        if not enabled:
            return

        # Try pyttsx3 first (best for Windows — offline,
        # no file writing, instant)
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 165)
            self.engine.setProperty('volume', 0.95)

            # Pick a clear English voice if available
            voices = self.engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or \
                   'zira' in v.name.lower() or \
                   'david' in v.name.lower():
                    self.engine.setProperty('voice', v.id)
                    break

            self._use_pyttsx3 = True
            print("✅ Voice coach: using pyttsx3 (Windows TTS)")

        except Exception as e:
            print(f"pyttsx3 not available ({e}), "
                  f"trying gTTS fallback...")
            self._use_pyttsx3 = False

        self._start_worker()

    def _start_worker(self):
        t = threading.Thread(
            target=self._worker, daemon=True
        )
        t.start()

    def _worker(self):
        while True:
            try:
                priority, _, msg = self.msg_queue.get(
                    timeout=1
                )
                self._speak(msg)
                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice worker error: {e}")

    def _speak(self, msg: CoachMessage):
        if not self.enabled:
            return

        self.speaking = True
        try:
            if self._use_pyttsx3 and self.engine:
                self._speak_pyttsx3(msg.text)
            else:
                self._speak_gtts(msg.text, msg.category)
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.speaking = False

    def _speak_pyttsx3(self, text: str):
        """
        pyttsx3 is not thread-safe so we
        reinit per call on Windows.
        """
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty('rate', 165)
            eng.setProperty('volume', 0.95)
            eng.say(text)
            eng.runAndWait()
            eng.stop()
        except Exception as e:
            print(f"pyttsx3 speak error: {e}")

    def _speak_gtts(self, text: str, category: str):
        """gTTS + playsound fallback."""
        try:
            from gtts import gTTS
            path = os.path.join(
                self.audio_dir,
                f"coach_{category}.mp3"
            )
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(path)

            # Windows: use playsound
            try:
                from playsound import playsound
                playsound(path, block=True)
            except Exception:
                # Last resort: Windows built-in
                import subprocess
                subprocess.Popen(
                    ['start', '/min', '', path],
                    shell=True
                )
                time.sleep(3)

        except Exception as e:
            print(f"gTTS speak error: {e}")

    def say(self, text: str,
            category: str = 'form',
            priority: int = 2):
        """Queue a message with cooldown protection."""
        if not self.enabled:
            return

        now      = time.time()
        cooldown = self.COOLDOWNS.get(category, 8.0)
        last     = self.last_spoken.get(category, 0)

        if now - last < cooldown:
            return

        self.last_spoken[category] = now
        msg = CoachMessage(
            text=text,
            priority=priority,
            category=category
        )
        # Lower number = higher priority in PriorityQueue
        import time as _time
        self.msg_queue.put((-priority, _time.time(), msg))

    def analyze_and_coach(
        self,
        fatigue_score:     float,
        performance_score: float,
        movement_quality:  str,
        injury_risk:       float,
        cadence:           float,
        stride_regularity: float,
        fatigue_threshold: int = 70,
        perf_threshold:    int = 30
    ):
        """
        Fires the right voice cue automatically
        based on all current metrics.
        """

        # ── Fatigue ──
        if fatigue_score > 85:
            self.say(
                "Critical fatigue. Consider slowing down.",
                category='fatigue', priority=3
            )
        elif fatigue_score > fatigue_threshold:
            self.say(
                f"Fatigue at {fatigue_score:.0f} percent. "
                "Focus on your breathing.",
                category='fatigue', priority=2
            )

        # ── Movement quality ──
        if movement_quality == 'Critical':
            self.say(
                "Movement quality critical. "
                "Check your knee and hip alignment.",
                category='form', priority=3
            )
        elif movement_quality == 'Degraded':
            self.say(
                "Form is degrading. "
                "Stay upright and drive your arms.",
                category='form', priority=2
            )

        # ── Injury risk ──
        if injury_risk > 70:
            self.say(
                "High injury risk. "
                "Asymmetry detected in your stride.",
                category='injury', priority=3
            )
        elif injury_risk > 45:
            self.say(
                "Moderate injury risk. "
                "Watch your knee alignment.",
                category='injury', priority=2
            )

        # ── Cadence ──
        if 0 < cadence < 120:
            self.say(
                "Increase cadence. "
                "Take shorter, quicker steps.",
                category='pace', priority=1
            )
        elif cadence > 185:
            self.say(
                "Excellent sprint cadence. Keep it up!",
                category='pace', priority=1
            )

        # ── Stride regularity ──
        if 0 < stride_regularity < 0.5:
            self.say(
                "Stride pattern uneven. "
                "Focus on rhythm.",
                category='form', priority=2
            )

        # ── Performance drop ──
        if performance_score < perf_threshold:
            self.say(
                "Performance dropping. "
                "Engage your core and push through.",
                category='form', priority=2
            )

    def test(self):
        """Quick test — call this to verify voice works."""
        self.say(
            "Voice coach is active and working.",
            category='form',
            priority=3
        )
        time.sleep(4)

    def stop(self):
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass