
# microscope_live_safe.py
import time
import cv2
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()

# Use preview configuration for live view
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Start with Auto Exposure ON to get a visible image
picam2.set_controls({"AeEnable": True})
print("Auto exposure enabled. Press m to switch to manual.")

def draw_overlay(img):
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w//2, h//2), color=(0, 255, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
    cv2.putText(img, "Microscope Live (c=save, m=manual AE off, +/- gain, [ ] exposure, q=quit)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def save_image(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"microscope_{timestamp}.png"
    cv2.imwrite(fname, frame)
    print(f"Saved: {fname}")

manual = False
exposure = 30000  # start brighter for microscopy
gain = 4.0

while True:
    frame = picam2.capture_array()
    # Print quick stats occasionally
    if int(time.time()) % 5 == 0:
        print(f"Stats: min={frame.min()} max={frame.max()} mean={frame.mean():.1f}")

    display = frame.copy()
    draw_overlay(display)

    try:
        cv2.imshow("Microscope", display)
    except cv2.error:
        # Headless environment; skip showing
        pass

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('c'):
        save_image(frame)
    elif k == ord('m'):
        manual = not manual
        if manual:
            picam2.set_controls({"AeEnable": False, "ExposureTime": exposure, "AnalogueGain": gain})
            print(f"Manual mode ON: Exposure={exposure}us, Gain={gain}x")
        else:
            picam2.set_controls({"AeEnable": True})
            print("Auto exposure enabled.")
    elif k == ord('+') and manual:
        gain = min(gain + 0.5, 16.0)
        picam2.set_controls({"AnalogueGain": gain})
        print(f"Gain: {gain:.1f}x")
    elif k == ord('-') and manual:
        gain = max(gain - 0.5, 1.0)
        picam2.set_controls({"AnalogueGain": gain})
        print(f"Gain: {gain:.1f}x")
    elif k == ord(']') and manual:
        exposure = min(exposure + 5000, 100000)  # up to 100 ms
        picam2.set_controls({"ExposureTime": exposure})
        print(f"Exposure: {exposure} us")
    elif k == ord('[') and manual:
        exposure = max(exposure - 5000, 1000)
        picam2.set_controls({"ExposureTime": exposure})
        print(f"Exposure: {exposure} us")

cv2.destroyAllWindows()
picam2.stop()
