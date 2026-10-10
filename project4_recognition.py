"""
DecodeLabs AI - Project 4: Image / Text Recognition (Basic)

Two execution paths, matching the project brief:
  Path 1 - OCR (pytesseract): grayscale -> blur -> threshold -> deskew -> read text
  Path 2 - Object detection (cv2.dnn + MobileNet-SSD): blob -> forward pass -> boxes

Both paths enforce the 80% confidence gate and save a visual output.

Usage:
  python project4_recognition.py ocr    input.png
  python project4_recognition.py detect input.jpg  --proto deploy.prototxt --model mobilenet_iter_73000.caffemodel
  python project4_recognition.py ocr    --demo          # builds a sample image and runs on it
"""
import argparse
import sys

import cv2
import numpy as np
import pytesseract

CONF_THRESHOLD = 0.80  # the "80% gate" from the brief

# 20 classes MobileNet-SSD (Caffe, VOC) was trained on
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]


# ----------------------------------------------------------------------------
# Pre-processing (Logic Skeleton)
# ----------------------------------------------------------------------------
def deskew(binary):
    """Estimate text rotation from foreground pixels and rotate to horizontal."""
    coords = np.column_stack(np.where(binary < 128))  # dark (text) pixels
    if len(coords) < 50:
        return binary
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3:  # already straight
        return binary
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess(image):
    """Step 1 grayscale, Step 2 Gaussian blur, Step 3 threshold + deskew."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)              # 3D RGB -> 1D intensity
    blur = cv2.GaussianBlur(gray, (5, 5), 0)                     # remove micro-noise
    # Adaptive thresholding handles uneven lighting; Otsu picks a global cutoff.
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu suits evenly lit scans; adaptive suits shadows / uneven lighting.
    # Pick adaptive when brightness varies a lot across the image.
    uneven = cv2.GaussianBlur(gray, (101, 101), 0).std() > 25
    binary = adaptive if uneven else otsu
    return gray, deskew(binary)


# ----------------------------------------------------------------------------
# Path 1: OCR
# ----------------------------------------------------------------------------
def run_ocr(image, psm=3):
    gray, binary = preprocess(image)
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(binary, config=config,
                                     output_type=pytesseract.Output.DICT)
    out = image.copy()
    kept, dropped = [], 0
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        conf /= 100.0
        if conf >= CONF_THRESHOLD:                      # IF confidence >= 0.80
            x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 160, 0), 2)
            cv2.putText(out, f"{word} {conf:.0%}", (x, max(y - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
            kept.append((word, conf))
        else:                                           # ELSE drop_detection()
            dropped += 1
    return kept, dropped, binary, out


# ----------------------------------------------------------------------------
# Path 2: Object detection (MobileNet-SSD)
# ----------------------------------------------------------------------------
def run_detection(image, proto, model):
    net = cv2.dnn.readNetFromCaffe(proto, model)
    h, w = image.shape[:2]
    # 4D blob: resize to 300x300, mean subtraction (127.5), scale 1/127.5
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()                           # shape (1, 1, N, 7)
    out = image.copy()
    kept, dropped = [], 0
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        idx = int(detections[0, 0, i, 1])
        if conf >= CONF_THRESHOLD:
            # Normalized (0-1) coords -> real pixel coords
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            label = f"{VOC_CLASSES[idx]}: {conf:.0%}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            kept.append((VOC_CLASSES[idx], conf))
        elif conf > 0.05:
            dropped += 1
    return kept, dropped, out


# ----------------------------------------------------------------------------
def make_demo_image(path="demo_input.png"):
    """Create a slightly tilted, noisy sample so the pipeline has something to read."""
    img = np.full((300, 800, 3), 235, np.uint8)
    cv2.putText(img, "DecodeLabs Project 4", (40, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (30, 30, 30), 4)
    cv2.putText(img, "Image and Text Recognition", (40, 220), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (30, 30, 30), 3)
    M = cv2.getRotationMatrix2D((400, 150), 3, 1.0)
    img = cv2.warpAffine(img, M, (800, 300), borderValue=(235, 235, 235))
    noise = np.random.default_rng(0).normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ocr", "detect"])
    ap.add_argument("image", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
    ap.add_argument("--proto")
    ap.add_argument("--model")
    ap.add_argument("--out", default="output.png")
    a = ap.parse_args()

    path = make_demo_image() if a.demo else a.image
    if not path:
        sys.exit("Provide an image path or use --demo")
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read image: {path}")

    if a.mode == "ocr":
        kept, dropped, binary, out = run_ocr(image, a.psm)
        cv2.imwrite("preprocessed.png", binary)
        print("Recognized text (>= 80% confidence):")
        print(" ".join(w for w, _ in kept) or "(nothing passed the gate)")
    else:
        if not (a.proto and a.model):
            sys.exit("detect mode needs --proto and --model")
        kept, dropped, out = run_detection(image, a.proto, a.model)
        print("Detections (>= 80% confidence):")

    for label, conf in kept:
        print(f"  {label:<20} {conf:.1%}")
    print(f"Accepted: {len(kept)} | Dropped below 80%: {dropped}")
    if kept:
        print(f"Lowest accepted confidence: {min(c for _, c in kept):.1%}  -> milestone 3 "
              f"{'PASSED' if min(c for _, c in kept) >= CONF_THRESHOLD else 'FAILED'}")
    cv2.imwrite(a.out, out)
    print(f"Visual output saved to {a.out}")


if __name__ == "__main__":
    main()
