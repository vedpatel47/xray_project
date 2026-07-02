"""
=============================================================================
X-RAY CLAIM VALIDATOR — Gradio Demo
=============================================================================
Cotiviti Intern Assessment — Topic 1: Clinical NLP / Computer Vision POC

WHAT THIS DOES
--------------
1. Takes a chest X-ray image as input
2. Runs it through the trained CNN (MobileNetV2) to predict NORMAL / PNEUMONIA
3. Cross-checks that imaging finding against billed ICD-10 diagnosis codes
4. Returns SUPPORTED / CONTRADICTED / UNSUPPORTED verdict per code
5. Assigns an overall claim risk level: LOW / MEDIUM / HIGH

This directly mirrors Cotiviti's prepay claim review workflow:
    "Does the imaging documentation support what was billed?"

HOW TO RUN
----------
Option A — Google Colab (recommended for demo):
    1. Upload this file + xray_classifier.keras to Colab
    2. Run:
        !pip install gradio tensorflow pillow -q
        !python app.py
    3. Click the public ngrok URL that Gradio prints

Option B — Local machine:
    pip install gradio tensorflow pillow
    python app.py
    Open http://localhost:7860

REQUIREMENTS
------------
- xray_classifier.keras  (output of train_model.py, must be in same folder)
- pip: gradio tensorflow pillow numpy
=============================================================================
"""

import os
import json
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# =============================================================================
# 1. ICD-10 Code Registry
#    Maps code → (description, imaging_relevant_finding)
#    imaging_relevant_finding is what the CNN needs to see to SUPPORT this code
# =============================================================================

ICD10_REGISTRY = {
    "J18.9": {
        "description": "Pneumonia, unspecified organism",
        "requires_finding": "PNEUMONIA",
        "note": "Requires active pneumonia findings on chest imaging",
    },
    "J18.1": {
        "description": "Lobar pneumonia, unspecified organism",
        "requires_finding": "PNEUMONIA",
        "note": "Requires lobar consolidation pattern on chest imaging",
    },
    "J15.9": {
        "description": "Unspecified bacterial pneumonia",
        "requires_finding": "PNEUMONIA",
        "note": "Requires infiltrate or consolidation on chest imaging",
    },
    "I10": {
        "description": "Essential (primary) hypertension",
        "requires_finding": None,   # not detectable on chest X-ray alone
        "note": "Not directly verifiable on chest X-ray (requires clinical/lab data)",
    },
    "E11.9": {
        "description": "Type 2 diabetes mellitus without complications",
        "requires_finding": None,
        "note": "Not detectable on chest X-ray",
    },
    "J44.9": {
        "description": "Chronic obstructive pulmonary disease, unspecified",
        "requires_finding": None,   # COPD needs PFT + clinical, not just X-ray
        "note": "Chest X-ray findings alone insufficient for COPD diagnosis",
    },
    "J96.00": {
        "description": "Acute respiratory failure, unspecified",
        "requires_finding": "PNEUMONIA",   # pneumonia is a common trigger
        "note": "Respiratory failure secondary to pneumonia is supported by imaging",
    },
    "Z87.01": {
        "description": "Personal history of pneumonia",
        "requires_finding": "NORMAL",   # history code — imaging should be clear now
        "note": "History code — current imaging should show resolved/normal findings",
    },
}

COMMON_CODES = list(ICD10_REGISTRY.keys())

# =============================================================================
# 2. Load the trained model
# =============================================================================

MODEL_PATH = "xray_classifier.keras"
IMG_SIZE   = (224, 224)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found.\n"
            "Please run train_model.py first to generate the model,\n"
            "then place xray_classifier.keras in the same folder as this file."
        )
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model

try:
    model = load_model()
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    print("Running in DEMO MODE — predictions will be simulated.")
    model = None
    MODEL_LOADED = False


# =============================================================================
# 3. Prediction logic
# =============================================================================

def predict_xray(pil_image, confidence_threshold=0.5):
    """
    Run the CNN on a PIL image and return prediction + confidence.
    Falls back to a simulated prediction if model not loaded (for testing UI).
    """
    if not MODEL_LOADED:
        # Demo mode — simulate a pneumonia prediction for UI testing
        print("DEMO MODE: simulating prediction")
        return "PNEUMONIA", 0.87, 0.87

    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw_prob  = float(model.predict(img_array, verbose=0)[0][0])
    label     = "PNEUMONIA" if raw_prob > confidence_threshold else "NORMAL"
    confidence = raw_prob if label == "PNEUMONIA" else 1.0 - raw_prob

    return label, confidence, raw_prob


def reconcile_codes(imaging_finding, confidence, selected_codes):
    """
    Core claim validation logic.
    For each billed code, determine if imaging supports, contradicts,
    or cannot assess the diagnosis.
    """
    verdicts = []

    for code in selected_codes:
        info = ICD10_REGISTRY.get(code)
        if not info:
            verdicts.append({
                "code": code,
                "description": "Unknown code",
                "status": "UNKNOWN",
                "rationale": "Code not found in registry.",
                "emoji": "❓",
            })
            continue

        required = info["requires_finding"]

        if required is None:
            # Code not assessable by chest X-ray imaging alone
            status   = "NOT ASSESSABLE BY IMAGING"
            rationale = (
                f"{info['note']} This code requires clinical/lab data "
                "beyond what chest X-ray can provide."
            )
            emoji = "⚠️"

        elif required == imaging_finding:
            # Imaging finding matches what the code requires
            status    = "SUPPORTED"
            rationale = (
                f"Imaging finding ({imaging_finding}, {confidence:.1%} confidence) "
                f"is consistent with billed diagnosis. {info['note']}"
            )
            emoji = "✅"

        else:
            # Imaging explicitly contradicts the billed code
            status    = "CONTRADICTED"
            rationale = (
                f"Imaging shows {imaging_finding} ({confidence:.1%} confidence) "
                f"but billed code requires {required}. "
                f"{info['note']} — HIGH overpayment risk."
            )
            emoji = "🚨"

        verdicts.append({
            "code": code,
            "description": info["description"],
            "status": status,
            "rationale": rationale,
            "emoji": emoji,
        })

    return verdicts


def compute_risk_level(verdicts):
    statuses = [v["status"] for v in verdicts]
    if "CONTRADICTED" in statuses:
        return "🔴 HIGH", "One or more billed codes are directly contradicted by imaging findings."
    elif "SUPPORTED" not in statuses:
        return "🟡 MEDIUM", "No billed codes are supported by imaging. Manual review recommended."
    elif "NOT ASSESSABLE BY IMAGING" in statuses:
        return "🟡 MEDIUM", "Some codes require additional clinical data beyond imaging."
    else:
        return "🟢 LOW", "All imaging-assessable codes are supported by documentation."


# =============================================================================
# 4. Gradio interface
# =============================================================================

def validate_claim(
    xray_image,
    selected_codes,
    confidence_threshold,
):
    """Main function called by Gradio on every submission."""

    if xray_image is None:
        return (
            "⚠️ Please upload a chest X-ray image.",
            "", "", "", ""
        )

    if not selected_codes:
        return (
            "⚠️ Please select at least one ICD-10 code to validate.",
            "", "", "", ""
        )

    # --- Step 1: Run CNN prediction ---
    pil_image = Image.fromarray(xray_image) if isinstance(xray_image, np.ndarray) else xray_image
    imaging_finding, confidence, raw_prob = predict_xray(pil_image, confidence_threshold)

    # --- Step 2: Reconcile against billed codes ---
    verdicts = reconcile_codes(imaging_finding, confidence, selected_codes)

    # --- Step 3: Compute overall claim risk ---
    risk_level, risk_explanation = compute_risk_level(verdicts)

    # --- Format outputs ---

    # Imaging result panel
    imaging_result = (
        f"## Imaging Finding\n\n"
        f"**{'🫁 PNEUMONIA DETECTED' if imaging_finding == 'PNEUMONIA' else '✅ NORMAL — No Pneumonia Detected'}**\n\n"
        f"Confidence: **{confidence:.1%}**  \n"
        f"Raw probability (PNEUMONIA): {raw_prob:.4f}  \n"
        f"Threshold used: {confidence_threshold:.2f}\n\n"
        f"{'> ⚕️ Imaging suggests active pulmonary infiltrate or consolidation consistent with pneumonia.' if imaging_finding == 'PNEUMONIA' else '> ⚕️ No significant pulmonary infiltrate detected. Lung fields appear clear.'}"
    )

    # Code verdict panel
    verdict_lines = ["## Code-by-Code Validation\n"]
    for v in verdicts:
        verdict_lines.append(
            f"### {v['emoji']} `{v['code']}` — {v['description']}\n"
            f"**Status:** {v['status']}  \n"
            f"**Rationale:** {v['rationale']}\n"
        )
    verdicts_result = "\n".join(verdict_lines)

    # Risk summary panel
    risk_result = (
        f"## Claim Risk Assessment\n\n"
        f"**Overall Risk Level: {risk_level}**\n\n"
        f"{risk_explanation}\n\n"
        f"---\n"
        f"*This assessment is based on imaging evidence only. A complete "
        f"payment integrity review also considers clinical notes, lab results, "
        f"and provider documentation. This tool demonstrates the imaging "
        f"component of Cotiviti's multi-modal claim validation pipeline.*"
    )

    # JSON output for technical view
    json_output = json.dumps({
        "imaging_finding": imaging_finding,
        "confidence": round(confidence, 4),
        "raw_probability_pneumonia": round(raw_prob, 4),
        "claim_risk_level": risk_level.split()[-1],  # LOW / MEDIUM / HIGH
        "verdicts": verdicts,
    }, indent=2)

    return imaging_result, verdicts_result, risk_result, json_output


# =============================================================================
# 5. Build and launch the Gradio UI
# =============================================================================

DESCRIPTION = """
## X-Ray Claim Validator
### Clinical Computer Vision for Payment Integrity

This tool demonstrates how AI-powered chest X-ray analysis can be integrated
into a **payment integrity workflow** — verifying that billed ICD-10 diagnosis
codes are supported by imaging documentation before a claim is paid.

**How to use:**
1. Upload a chest X-ray image (PA view works best)
2. Select the ICD-10 codes billed on the claim
3. Click **Validate Claim** to see the reconciliation verdict

*Built on MobileNetV2 fine-tuned on the Kaggle Chest X-Ray Pneumonia dataset
(5,863 images). For educational and demonstration purposes only.*
"""

SAMPLE_NOTE = """
**Example use case:**  
A provider submits a claim for a 64-year-old patient with:
- `J18.9` Pneumonia  
- `I10` Hypertension  

The chest X-ray shows **no infiltrate** (NORMAL finding).  
The validator flags `J18.9` as **CONTRADICTED** → HIGH risk → claim held for review.
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    title="X-Ray Claim Validator | Cotiviti POC",
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")

            xray_input = gr.Image(
                label="Chest X-Ray Image",
                type="pil",
                height=300,
                sources=["upload", "clipboard"],
            )

            code_selector = gr.CheckboxGroup(
                choices=[
                    f"{code} — {ICD10_REGISTRY[code]['description']}"
                    for code in COMMON_CODES
                ],
                label="Billed ICD-10 Codes (select all that appear on the claim)",
                value=[
                    f"J18.9 — {ICD10_REGISTRY['J18.9']['description']}",
                    f"I10 — {ICD10_REGISTRY['I10']['description']}",
                ],
            )

            threshold_slider = gr.Slider(
                minimum=0.3,
                maximum=0.8,
                value=0.5,
                step=0.05,
                label="Prediction Threshold (lower = more sensitive to pneumonia)",
                info="0.5 is standard; lower for higher sensitivity, higher for higher specificity",
            )

            submit_btn = gr.Button(
                "🔍 Validate Claim",
                variant="primary",
                size="lg",
            )

            gr.Markdown(SAMPLE_NOTE)

        # Right column — outputs
        with gr.Column(scale=1):
            gr.Markdown("### Results")

            imaging_output = gr.Markdown(label="Imaging Analysis")

            with gr.Accordion("Code-by-Code Verdicts", open=True):
                verdicts_output = gr.Markdown()

            risk_output = gr.Markdown(label="Claim Risk")

            with gr.Accordion("Raw JSON Output (for developers)", open=False):
                json_output = gr.Code(language="json", label="Structured Output")

    # Wire up the button
    submit_btn.click(
        fn=lambda img, codes, threshold: validate_claim(
            img,
            # extract just the code part (before " — ")
            [c.split(" — ")[0] for c in codes],
            threshold,
        ),
        inputs=[xray_input, code_selector, threshold_slider],
        outputs=[imaging_output, verdicts_output, risk_output, json_output],
    )

    gr.Markdown(
        "---\n"
        "*Cotiviti Intern Assessment — Topic 1: Clinical NLP / Computer Vision POC  |  "
        "Model: MobileNetV2 fine-tuned on Kaggle Chest X-Ray Pneumonia dataset*"
    )


if __name__ == "__main__":
    demo.launch(
        share=True,          # generates a public URL — needed for Colab
        server_port=7860,
        show_error=True,
    )
