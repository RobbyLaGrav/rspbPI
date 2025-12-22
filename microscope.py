
# microscope_live.py
import time
import cv2
import numpy as np
from picamera2 import Picamera2, Preview

# Initialize camera
picam2 = Picamera2()

# Configure for preview (choose your resolution)
config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()

# Set initial exposure/gain
controls = {
    "AeEnable": False,           # manual exposure
    "ExposureTime": 8000,        # microseconds (8 ms)
    "AnalogueGain": 1.0          # 1x gain
}
picam2.set_controls({"AeEnable": True})

print("Controls: c=capture, +/-=gain, [ ]=exposure, h=histogram, q=quit")

def draw_overlay(img):
    # Crosshair + scale bar placeholder
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w//2, h//2), color=(0, 255, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
    # Add simple text overlay
    cv2.putText(img, "Microscope Live (c=save, +/- gain, [ ] exposure, a=autofocus, q=quit)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def save_image(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"microscope_{timestamp}.png"
    cv2.imwrite(fname, frame)
    print(f"Saved: {fname}")

def focus_metric(img_gray):
    # Use variance of Laplacian as a contrast focus metric
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

# Live loop
exposure = controls["ExposureTime"]
gain = controls["AnalogueGain"]

while True:
    frame = picam2.capture_array()
    display = frame.copy()
    draw_overlay(display)
    cv2.imshow("Microscope", display)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('c'):
        save_image(frame)
    elif k == ord('+'):
        gain = min(gain + 0.1, 8.0)   # cap gain
        picam2.set_controls({"AnalogueGain": gain})
        print(f"Gain: {gain:.2f}x")
    elif k == ord('-'):
        gain = max(gain - 0.1, 1.0)
        picam2.set_controls({"AnalogueGain": gain})
        print(f"Gain: {gain:.2f}x")
    elif k == ord(']'):
        exposure = min(exposure + 1000, 80000)  # up to 80 ms
        picam2.set_controls({"ExposureTime": exposure})
        print(f"Exposure: {exposure} us")
    elif k == ord('['):
        exposure = max(exposure - 1000, 1000)
        picam2.set_controls({"ExposureTime": exposure})
        print(f"Exposure: {exposure} us")
    elif k == ord('h'):
        # Show histogram for exposure tuning
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        print("Histogram peaks (top 5 bins):", np.argsort(hist.ravel())[::-1][:5])
    elif k == ord('a'):
        # Basic autofocus: you move focus manually; script suggests best exposure for contrast
        # (Full motor autofocus provided in section 5 below)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        m = focus_metric(gray)
        print(f"Focus metric (Laplacian var): {m:.2f}")

cv2.destroyAllWindows()
picam2.stop()
``
