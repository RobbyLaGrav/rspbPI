import cv2
import time
import json
import numpy as np
from picamera2 import Picamera2

# ================= CONFIG =================
WIDTH, HEIGHT = 1280, 720
WINDOW = "Ultimate Real Microscope"
CAL_FILE = "calibration.json"

Z_MIN, Z_MAX, Z_STEPS = 3.0, 8.0, 25

focus_pos = 5.0
exposure = 30000
gain = 4.0
manual_ae = False
recording = False
video_writer = None

status = ""
status_until = 0

# ================= CAMERA =================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
)
picam2.start()
picam2.set_controls({"AeEnable": True})

AF_SUPPORTED = True
try:
    picam2.set_controls({"AfMode": 0, "LensPosition": focus_pos})
except:
    AF_SUPPORTED = False

# ================= UTIL =================
def show(msg, sec=2):
    global status, status_until
    status = msg
    status_until = time.time() + sec

def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def load_cal():
    try:
        return json.load(open(CAL_FILE))["um_px"]
    except:
        return None

def save_cal(um_px):
    json.dump({"um_px": um_px}, open(CAL_FILE, "w"))
    show("Calibration saved")

# ================= REAL AUTOFOCUS =================
def autofocus():
    global focus_pos
    best_z, best_s = focus_pos, 0
    picam2.set_controls({"AeEnable": False})

    for z in np.linspace(Z_MIN, Z_MAX, Z_STEPS):
        picam2.set_controls({"LensPosition": z})
        time.sleep(0.05)
        f = picam2.capture_array()
        s = sharpness(f)
        if s > best_s:
            best_s, best_z = s, z

    focus_pos = best_z
    picam2.set_controls({
        "LensPosition": focus_pos,
        "AeEnable": not manual_ae
    })
    show(f"Focused @ {focus_pos:.2f}")

# ================= REAL Z-STACK =================
def capture_zstack():
    imgs = []
    for z in np.linspace(Z_MIN, Z_MAX, Z_STEPS):
        picam2.set_controls({"LensPosition": z})
        time.sleep(0.05)
        imgs.append(picam2.capture_array())
    show("Z-stack captured")
    return imgs

# ================= REAL FOCUS STACKING =================
def focus_stack(images):
    gray = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in images]
    lap = [cv2.Laplacian(g, cv2.CV_64F) for g in gray]
    lap_stack = np.stack(lap, axis=-1)
    idx = np.argmax(lap_stack, axis=-1)

    stacked = np.zeros_like(images[0])
    for y in range(stacked.shape[0]):
        for x in range(stacked.shape[1]):
            stacked[y, x] = images[idx[y, x]][y, x]

    cv2.imwrite("focus_stacked.png", stacked)
    show("Focus stack saved")

# ================= REAL CELL COUNT =================
def count_cells(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(g, (5,5), 0)
    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    show(f"Cells: {len(cnts)}")

# ================= REAL STITCHING =================
def stitch(images):
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        cv2.imwrite("stitched.png", pano)
        show("Stitched image saved")
    else:
        show("Stitch failed")

# ================= UI =================
def draw_ui(img, s):
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w//2, h//2), (0,255,0),
                   cv2.MARKER_CROSS, 30, 2)

    panel = np.zeros((170, 480, 3), dtype=np.uint8)
    panel[:] = (25,25,25)

    lines = [
        f"Focus {focus_pos:.2f}",
        f"Exposure {exposure} us",
        f"Gain {gain:.1f}x",
        f"Sharpness {int(s)}",
        f"AE {'MANUAL' if manual_ae else 'AUTO'}"
    ]

    for i, t in enumerate(lines):
        cv2.putText(panel, t, (10, 25+i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 1)

    img[10:180, 10:490] = panel

    if time.time() < status_until:
        cv2.putText(img, status, (w//2-200, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,255), 2)

    cv2.putText(img,
        "a=AF | z=Zstack | s=Stack | x=Cells | m=AE | <> focus | [] exp | +/- gain | q=quit",
        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

# ================= MAIN =================
cv2.namedWindow(WINDOW)

zstack_cache = None

while True:
    frame = picam2.capture_array()
    s = sharpness(frame)
    view = frame.copy()
    draw_ui(view, s)
    cv2.imshow(WINDOW, view)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('a'):
        autofocus()
    elif k == ord('z'):
        zstack_cache = capture_zstack()
    elif k == ord('s') and zstack_cache:
        focus_stack(zstack_cache)
    elif k == ord('x'):
        count_cells(frame)
    elif k == ord('<'):
        focus_pos = max(0, focus_pos - 0.1)
        picam2.set_controls({"LensPosition": focus_pos})
    elif k == ord('>'):
        focus_pos = min(10, focus_pos + 0.1)
        picam2.set_controls({"LensPosition": focus_pos})
    elif k == ord('m'):
        manual_ae = not manual_ae
        picam2.set_controls({"AeEnable": not manual_ae})
    elif k == ord('+'):
        gain = min(16, gain + 0.5)
        picam2.set_controls({"AnalogueGain": gain})
    elif k == ord('-'):
        gain = max(1, gain - 0.5)
        picam2.set_controls({"AnalogueGain": gain})
    elif k == ord(']'):
        exposure = min(100000, exposure + 5000)
        picam2.set_controls({"ExposureTime": exposure})
    elif k == ord('['):
        exposure = max(1000, exposure - 5000)
        picam2.set_controls({"ExposureTime": exposure})

cv2.destroyAllWindows()
picam2.stop()
