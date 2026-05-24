import subprocess
import sys
import os


def run(cmd):
    print(f"\n> {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr and 'error' in result.stderr.lower():
        print(f"STDERR: {result.stderr.strip()}")
    return result.returncode == 0


print("=" * 50)
print("Windows Setup & Diagnostics")
print("=" * 50)


# ── Step 1: Install Windows-specific packages ──
print("\n[1/4] Installing Windows audio packages...")
run("pip install pyttsx3 pywin32")
run("pip install gtts playsound==1.2.2")
run("pip install pygame scipy")


# ── Step 2: Test camera ──
print("\n[2/4] Testing camera...")
try:
    import cv2
    found = False
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"  Camera found at index {i} "
                      f"- shape: {frame.shape}")
                found = True
                break
    if not found:
        print("  No camera found! Check Device Manager.")
except Exception as e:
    print(f"  Camera test error: {e}")


# ── Step 3: Test pyttsx3 voice ──
print("\n[3/4] Testing voice (pyttsx3)...")
try:
    import pyttsx3
    eng = pyttsx3.init()
    voices = eng.getProperty('voices')
    print(f"  Found {len(voices)} voice(s):")
    for v in voices[:3]:
        print(f"    - {v.name}")
    eng.setProperty('rate', 165)
    eng.say("Voice coach test. System is working.")
    eng.runAndWait()
    eng.stop()
    print("  pyttsx3 voice: OK")
except Exception as e:
    print(f"  pyttsx3 error: {e}")
    print("  Trying gTTS fallback...")
    try:
        from gtts import gTTS
        os.makedirs("data/audio", exist_ok=True)
        tts = gTTS("Voice test", lang='en')
        tts.save("data/audio/test_voice.mp3")
        print("  gTTS: MP3 saved to data/audio/test_voice.mp3")
        print("  Manual check: open that file and see if audio plays")
    except Exception as e2:
        print(f"  gTTS error: {e2}")


# ── Step 4: Test MediaPipe ──
print("\n[4/4] Testing MediaPipe...")
try:
    import mediapipe as mp
    import numpy as np
    pose = mp.solutions.pose.Pose()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    pose.process(dummy)
    print("  MediaPipe: OK")
except Exception as e:
    print(f"  MediaPipe error: {e}")
    run("pip install mediapipe")


print("\n" + "=" * 50)
print("Setup complete.")
print("\nNow run:")
print("  python train.py        (if not done)")
print("  streamlit run app.py")
print("=" * 50)