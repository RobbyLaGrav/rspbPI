import cv2
import time
import json
import numpy as np
from picamera2 import Picamera2

# ==========================
# Configuration & Globals
# ==========================
WINDOW = "Ultimate Microscope"
CAL_FILE = "calibration.json"

focus_pos = 5.0
exposure = 30000
gain = 4.0
manual_ae = False
recording = False
video_writer = None
status_msg = ""
status_until = 0

Z_MIN, Z_MAX = 3.0, 8.0
Z_STEPS = 25

# ==========================
# Camera Setup
# ==========================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
)
picam2.start()

# Autofocus support
AF_SUPPORTED = True
try:
    picam2.set_controls({"AfMode": 0, "LensPosition": focus_pos})
except Exception:
    AF_SUPPORTED = False

picam2.set_controls({"AeEnable": True})

# ==========================
# Utility Functions
# ==========================
def show_status(msg, sec=2):
    global status_msg, status_until
    status_msg = msg
    status_until = time.time() + sec

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def save_image(img):
    name = f"img_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(name, img)
    show_status(f"Saved {name}")

def start_video(w, h):
    global video_writer, recording
    name = f"video_{time.strftime('%Y%m%d_%H%M%S')}.avi"
    video_writer = cv2.VideoWriter(
        name, cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h)
    )
    recording = True
    show_status("Recording started")

def stop_video():
    global recording
    video_writer.release()
    recording = False
    show_status("Recording stopped")

# ==========================
# Autofocus & Z-Stack
# ==========================
def autofocus_scan():
    global focus_pos
    show_status("Autofocus scan...")
    picam2.set_controls({"AeEnable": False})

    best_focus = focus_pos
    best_score = 0

    for z in np.linspace(Z_MIN, Z_MAX, Z_STEPS):
        picam2.set_controls({"LensPosition": z})
        time.sleep(0.05)
        frame = picam2.capture_array()
        s = sharpness(frame)
        if s > best_score:
            best_score = s
            best_focus = z

    focus_pos = best_focus
    picam2.set_controls({
        "LensPosition": focus_pos,
        "AeEnable": not manual_ae
    })
    show_status(f"Focus locked {focus_pos:.2f}")

def capture_zstack():
    show_status("Capturing Z-stack...")
    images = []
    for z in np.linspace(Z_MIN, Z_MAX, Z_STEPS):
        picam2.set_controls({"LensPosition": z})
        time.sleep(0.05)
        images.append(picam2.capture_array())
    show_status("Z-stack complete")
    return images

# ==========================
# Calibration & Measurement
# ==========================
def save_calibration(um_per_px):
    with open(CAL_FILE, "w") as f:
        json.dump({"um_px": um_per_px}, f)
    show_status("Calibration saved")

def load_calibration():
    try:
        with open(CAL_FILE) as f:
            return json.load(f)["um_px"]
    except:
        return None

def measure_distance(p1, p2):
    um_px = load_calibration()
    if um_px is None:
        show_status("No calibration")
        return None
    dist_px = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist_px * um_px

# ==========================
# Cell Counting (Lightweight CV)
# ==========================
def count_cells(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cnts, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return len(cnts)

# ==========================
# Motorized Stage (Stub)
# ==========================
def stage_move(dx, dy):
    print(f"Stage move dx={dx} dy={dy}")

# ==========================
# UI Overlay
# ==========================
def draw_ui(img, sharp):
    h, w = img.shape[:2]

    cv2.drawMarker(img, (w//2, h//2),
                   (0,255,0), cv2.MARKER_CROSS, 30, 2)

    panel = np.zeros((180, 480, 3), dtype=np.uint8)
    panel[:] = (30,30,30)

    lines = [
        f"Focus: {focus_pos:.2f}",
        f"Exposure: {exposure} us",
        f"Gain: {gain:.1f}x",
        f"Sharpness: {int(sharp)}",
        f"AE: {'MANUAL' if manual_ae else 'AUTO'}",
        f"REC: {'ON' if recording else 'OFF'}"
    ]

    for i, l in enumerate(lines):
        cv2.putText(panel, l, (10, 25 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 1)

    bar = int(min(sharp / 500, 1.0) * 460)
    cv2.rectangle(panel, (10, 160), (10 + bar, 170),
                  (0,200,0), -1)

    img[10:190, 10:490] = panel

    if time.time() < status_until:
        cv2.putText(img, status_msg,
                    (w//2 - 200, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,255), 2)

    cv2.putText(
        img,
        "a=AF scan | f=AF | z=Zstack | m=AE | <> focus | [] exp | +/- gain | c=cap | v=vid | x=cells | q=quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255,255,255), 1
    )

# ==========================
# Main Loop
# ==========================
cv2.namedWindow(WINDOW)

while True:
    frame = picam2.capture_array()
    sharp = sharpness(frame)

    if recording:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    display = frame.copy()
    draw_ui(display, sharp)

    cv2.imshow(WINDOW, display)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

    elif k == ord('c'):
        save_image(frame)

    elif k == ord('v'):
        if not recording:
            start_video(frame.shape[1], frame.shape[0])
        else:
            stop_video()

    elif k == ord('a'):
        autofocus_scan()

    elif k == ord('f') and AF_SUPPORTED:
        picam2.set_controls({"AfMode": 1})
        show_status("Hardware AF")

    elif k == ord('z'):
        capture_zstack()

    elif k == ord('<'):
        focus_pos = max(focus_pos - 0.1, 0)
        picam2.set_controls({"LensPosition": focus_pos})

    elif k == ord('>'):
        focus_pos = min(focus_pos + 0.1, 10)
        picam2.set_controls({"LensPosition": focus_pos})

    elif k == ord('m'):
        manual_ae = not manual_ae
        picam2.set_controls({"AeEnable": not manual_ae})

    elif k == ord('+'):
        gain = min(gain + 0.5, 16)
        picam2.set_controls({"AnalogueGain": gain})

    elif k == ord('-'):
        gain = max(gain - 0.5, 1)
        picam2.set_controls({"AnalogueGain": gain})

    elif k == ord(']'):
        exposure = min(exposure + 5000, 100000)
        picam2.set_controls({"ExposureTime": exposure})

    elif k == ord('['):
        exposure = max(exposure - 5000, 1000)
        picam2.set_controls({"ExposureTime": exposure})

    elif k == ord('x'):
        n = count_cells(frame)
        show_status(f"Cells detected: {n}")

# ==========================
# Cleanup
# ==========================
if recording:
    stop_video()
cv2.destroyAllWindows()
picam2.stop()
