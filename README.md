# X-Ray Claim Validator — POC
## Cotiviti Intern Assessment, Topic 1: Clinical NLP / Computer Vision

---

## What this project does

Takes a chest X-ray image + a list of billed ICD-10 codes and outputs whether each
code is SUPPORTED, CONTRADICTED, or NOT ASSESSABLE by the imaging findings.

This directly mirrors how Cotiviti's payment integrity workflow validates imaging
documentation against billed diagnoses before a claim is paid.

---

## Files

| File | Purpose |
|------|---------|
| `train_model.py` | Train MobileNetV2 on chest X-ray dataset (run in Colab) |
| `app.py` | Gradio demo UI — loads trained model and validates claims |
| `README.md` | This file |

After training, `train_model.py` saves:
- `xray_classifier.keras` — the trained model (put this next to `app.py`)
- `training_results.png` — confusion matrix + training curves

---

## Step-by-step: Run in Google Colab

### Step 1 — Open Colab and set GPU runtime
1. Go to https://colab.research.google.com
2. Runtime → Change runtime type → **GPU** (T4 free tier is enough)

### Step 2 — Get the dataset
Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
and download `chest-xray-pneumonia.zip` (~2GB).

**Upload to Colab** via Files panel (left sidebar → upload icon).
Then in a Colab cell:
```python
!unzip -q chest-xray-pneumonia.zip
```

### Step 3 — Upload train_model.py and run it
Upload `train_model.py` to Colab, then run each cell in order.
Training takes about 20-35 minutes on a T4 GPU.

Expected final metrics (approximate):
- Accuracy: ~92–95%
- AUC-ROC: ~0.96–0.98
- Recall (PNEUMONIA): ~95%+

### Step 4 — Download the trained model
After training, download `xray_classifier.keras` from the Colab Files panel.

### Step 5 — Run the Gradio demo
Upload `app.py` + `xray_classifier.keras` to a new Colab session (or same one).
Then run:
```python
!pip install gradio tensorflow pillow -q
!python app.py
```
Click the public URL Gradio prints — it works from any browser.

---

## Using the demo

1. Upload any chest X-ray image (PA view works best)
   - Test images are in `chest_xray/test/NORMAL/` and `chest_xray/test/PNEUMONIA/`
2. Check the ICD-10 codes you want to validate (pre-selected: J18.9 + I10)
3. Click **Validate Claim**
4. See the verdict: SUPPORTED / CONTRADICTED / NOT ASSESSABLE per code + overall claim risk

### Interesting test cases to show in your video

| Scenario | Image to use | Codes to select | Expected result |
|----------|-------------|-----------------|-----------------|
| Fully supported claim | PNEUMONIA test image | J18.9 | ✅ SUPPORTED — LOW risk |
| Contradicted code | NORMAL test image | J18.9 | 🚨 CONTRADICTED — HIGH risk |
| Mixed claim | PNEUMONIA test image | J18.9 + I10 | ✅ J18.9 supported, ⚠️ I10 not assessable by imaging |
| Completely wrong codes | NORMAL test image | E11.9 + I10 | ⚠️ Neither assessable — MEDIUM risk |

---

## Architecture

```
Chest X-Ray Image (224x224 RGB)
         ↓
MobileNetV2 (pretrained on ImageNet, fine-tuned on chest X-rays)
         ↓
GlobalAveragePooling → Dense(256) → Dropout → Dense(64) → Sigmoid
         ↓
PNEUMONIA probability [0.0 – 1.0]
         ↓
Threshold (default 0.5) → PNEUMONIA or NORMAL
         ↓
ICD-10 Reconciliation Engine
         ↓
SUPPORTED / CONTRADICTED / NOT ASSESSABLE per code
         ↓
Claim Risk Level: LOW / MEDIUM / HIGH
```

---

## Known limitations (mention these in your presentation — shows maturity)

1. **Binary classification only** — this model distinguishes NORMAL vs PNEUMONIA.
   A production system would classify many more conditions (atelectasis, pleural
   effusion, cardiomegaly, etc.) using a multi-label model like CheXNet.

2. **Imaging alone is insufficient for many codes** — hypertension (I10),
   diabetes (E11.9), and COPD (J44.9) cannot be confirmed by chest X-ray alone.
   The tool correctly flags these as NOT ASSESSABLE rather than inventing a verdict.

3. **No severity/specificity validation** — the model says "pneumonia present"
   but doesn't distinguish lobar vs. bronchopneumonia, which affects which specific
   code (J18.0 vs J18.1 vs J18.9) is appropriate.

4. **Dataset bias** — trained on one hospital system's images. Real production
   models need multi-institutional training data to generalize.

These are real limitations of every production radiology AI system too —
the paper you read from Tashkent Medical University discusses several of them.

---

## Why this connects to Cotiviti's business

Cotiviti's payment integrity products verify that billed codes are supported
by clinical documentation. This POC demonstrates that principle applied to
imaging: the CNN extracts a structured finding from an unstructured image
(the same way NLP extracts findings from unstructured text), and the
reconciliation engine then checks that finding against the claim — the
same SUPPORTED / CONTRADICTED / UNSUPPORTED logic from the text-based
Coding Support Validator, now extended to a second modality.
