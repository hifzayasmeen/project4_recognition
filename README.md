
# Project 4: Image & Text Recognition (Basic)

A Python pipeline that reads text from images (OCR) and detects objects using pre-trained models. Both paths enforce an **80% confidence gate** and save a labelled visual output.

Built as part of the **DecodeLabs AI Industrial Training Kit (Batch 2026)**.

---

## Features

- **Path 1: OCR** with `pytesseract` (Google's Tesseract engine)
- **Path 2: Object detection** with `cv2.dnn` and MobileNet-SSD
- **Image pre-processing:** grayscale, Gaussian blur, Otsu / adaptive thresholding, deskewing
- **Confidence filter:** results below 80% are dropped
- **Visual output:** bounding boxes with labels and confidence scores
- **Built-in demo mode** that generates a sample image, so no input file is needed

---

## How It Works

### Path 1: OCR

| Step | What it does |
|---|---|
| 1. Grayscale | Collapses the RGB image into a single intensity channel |
| 2. Gaussian blur | Removes small noise and imperfections |
| 3. Thresholding | Converts pixels to pure black or white (Otsu for even lighting, adaptive for shadows) |
| 4. Deskew | Measures text tilt and rotates it back to horizontal |
| 5. Tesseract | Reads words and returns a confidence score for each |

### Path 2: Object Detection

1. The image is converted to a 4D **blob** (resized to 300x300, mean subtracted, scaled) with `cv2.dnn.blobFromImage`.
2. MobileNet-SSD runs a single forward pass and returns normalized boxes, classes and confidences.
3. Coordinates are multiplied by the real image width and height to draw the boxes.

### The 80% Gate

```python
if confidence >= 0.80:
    draw_box_and_label()
else:
    drop_detection()
```

A model always outputs probabilities, so a threshold stops weak guesses from being reported as facts.

---

## Installation

**1. Install Tesseract OCR** (required for OCR mode)

- Windows: download the installer from the [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your PATH
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

**2. Install Python packages**

```bash
pip install opencv-python numpy pytesseract
```

> **Windows note:** if Python can't find Tesseract, add this line near the top of the script:
> `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`

---

## Usage

**Demo (no input file needed):**

```bash
python project4_recognition.py ocr --demo
```

**OCR on your own image:**

```bash
python project4_recognition.py ocr input.png
```

Choose a page layout with `--psm` (3 = mixed layouts, 6 = text block, 7 = single line, 11 = scattered text):

```bash
python project4_recognition.py ocr input.png --psm 6
```

**Object detection** (needs the MobileNet-SSD model files):

```bash
python project4_recognition.py detect input.jpg --proto deploy.prototxt --model mobilenet_iter_73000.caffemodel
```

**Options**

| Option | Description |
|---|---|
| `--demo` | Generate and use a sample image |
| `--psm N` | Tesseract page segmentation mode (OCR only) |
| `--proto`, `--model` | MobileNet-SSD files (detect only) |
| `--out FILE` | Output image name (default `output.png`) |

---

## Example Output

```
Recognized text (>= 80% confidence):
DecodeLabs Project 4 Image and Text Recognition
  DecodeLabs           92.0%
  Project              96.0%
  4                    96.0%
  Image                96.0%
  and                  96.0%
  Text                 96.0%
  Recognition          96.0%
Accepted: 7 | Dropped below 80%: 0
Lowest accepted confidence: 91.0%  -> milestone 3 PASSED
Visual output saved to output.png
```

---

## Project Milestones

- [x] **Library integration:** `pytesseract` and `cv2.dnn` implemented
- [x] **Pre-processing integrity:** grayscale conversion and thresholding
- [x] **Accuracy benchmark:** minimum 80% confidence on the final output
- [x] **Visual confirmation:** labelled output image generated

---

## Project Structure

```
project4_recognition/
├── project4_recognition.py   # main script (OCR + detection)
└── README.md
```

---

## Known Limitations

- Object detection needs the MobileNet-SSD `.prototxt` and `.caffemodel` files, which are not included in this repo.
- OCR boxes are measured on the deskewed image but drawn on the original, so they can sit slightly off heavily tilted text.
- MobileNet-SSD (VOC) recognizes only 20 object classes.

---

## Tech Stack

Python, OpenCV, Tesseract OCR (pytesseract), NumPy, MobileNet-SSD

---

## Author

**Hifza Yasmeen**
Built during the DecodeLabs AI training program.
