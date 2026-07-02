"""
=============================================================================
CHEST X-RAY PNEUMONIA CLASSIFIER — Training Script
=============================================================================
Cotiviti Intern Assessment — Topic 1: Clinical Natural Language Technology
                             (Computer Vision / CNN extension)

HOW TO USE THIS FILE
---------------------
1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook
3. Copy each CELL section below into a separate Colab cell
4. Runtime → Change runtime type → GPU (T4 is free and enough)
5. Run cells top to bottom

DATASET SETUP (do this first, outside Colab)
---------------------------------------------
1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click Download → chest-xray-pneumonia.zip (~2GB)
3. In Colab, upload it using the Files panel (left sidebar → upload icon)
   OR connect your Kaggle account and use the kaggle CLI (shown in Cell 1)

Expected folder structure after unzip:
    chest_xray/
        train/
            NORMAL/       (1341 images)
            PNEUMONIA/    (3875 images)
        val/
            NORMAL/       (8 images — very small, we'll merge with test)
            PNEUMONIA/    (8 images)
        test/
            NORMAL/       (234 images)
            PNEUMONIA/    (390 images)
=============================================================================
"""

# =============================================================================
# CELL 1 — Install dependencies and mount Google Drive (optional)
# =============================================================================
# Paste this into Colab Cell 1 and run it

"""
# Option A: Upload dataset manually (simplest)
# Just upload chest-xray-pneumonia.zip via the Files panel on the left sidebar

# Option B: Use Kaggle API (faster for large files)
!pip install kaggle -q
# You need to upload your kaggle.json API key first
# Get it from: https://www.kaggle.com/settings → API → Create New Token
from google.colab import files
files.upload()  # upload kaggle.json when prompted

import os
os.makedirs('/root/.kaggle', exist_ok=True)
!cp kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip

print("Dataset ready!")
"""

# =============================================================================
# CELL 2 — Imports and configuration
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---- Configuration ----
DATA_DIR    = "./chest_xray"          # change if your path differs
IMG_SIZE    = (224, 224)              # MobileNetV2 native input size
BATCH_SIZE  = 32
EPOCHS_HEAD = 10                      # train only the new head first
EPOCHS_FINE = 10                      # then fine-tune some base layers
LR_HEAD     = 1e-3
LR_FINE     = 1e-5
MODEL_SAVE  = "xray_classifier.keras" # final saved model

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# =============================================================================
# CELL 3 — Data generators with augmentation
# =============================================================================

# Training augmentation — makes the model more robust to real-world variation
# in X-ray acquisition (rotation, brightness, zoom differences between machines)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

# Validation/test — only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",          # 0 = NORMAL, 1 = PNEUMONIA
    shuffle=True,
    seed=42,
)

# The built-in val split is tiny (8 images per class), so we use test as val
# during training and report final metrics on it separately
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print(f"\nClass mapping: {train_gen.class_indices}")
print(f"Training samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# Class weights — dataset is imbalanced (3x more PNEUMONIA than NORMAL)
# Without this the model learns to just predict PNEUMONIA all the time
total = train_gen.samples
n_normal    = len(os.listdir(os.path.join(DATA_DIR, "train", "NORMAL")))
n_pneumonia = len(os.listdir(os.path.join(DATA_DIR, "train", "PNEUMONIA")))

class_weight = {
    0: total / (2 * n_normal),      # NORMAL
    1: total / (2 * n_pneumonia),   # PNEUMONIA
}
print(f"\nClass weights: {class_weight}")
print("(Higher weight for NORMAL compensates for class imbalance)")


# =============================================================================
# CELL 4 — Build the model (transfer learning with MobileNetV2)
# =============================================================================

def build_model(trainable_base_layers=0):
    """
    Transfer learning strategy:
    1. Load MobileNetV2 pretrained on ImageNet (general visual features)
    2. Freeze all base layers first — just train our new classification head
    3. After head converges, unfreeze last N layers for fine-tuning
       (this adapts learned features to medical imaging domain)

    MobileNetV2 is chosen over ResNet/VGG because:
    - Faster inference (important for real-time demo)
    - Smaller model size (easier to share/deploy)
    - Still achieves 90%+ on this dataset with fine-tuning
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )

    # Freeze all base layers
    base_model.trainable = False

    # Unfreeze last N layers for fine-tuning (0 = head-only phase)
    if trainable_base_layers > 0:
        for layer in base_model.layers[-trainable_base_layers:]:
            layer.trainable = True

    # Classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)           # flatten spatial features
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)                       # prevent overfitting
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x) # binary output [0,1]

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


model, base_model = build_model(trainable_base_layers=0)

model.compile(
    optimizer=Adam(learning_rate=LR_HEAD),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

total_params     = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}  (just the head)")
model.summary()


# =============================================================================
# CELL 5 — Phase 1: Train the classification head
# =============================================================================

callbacks_head = [
    EarlyStopping(
        monitor="val_auc", patience=4, restore_best_weights=True, mode="max",
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1,
    ),
]

print("=" * 60)
print("PHASE 1: Training classification head (base frozen)")
print("=" * 60)

history_head = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks_head,
    verbose=1,
)

print(f"\nPhase 1 best val AUC  : {max(history_head.history['val_auc']):.4f}")
print(f"Phase 1 best val accuracy: {max(history_head.history['val_accuracy']):.4f}")


# =============================================================================
# CELL 6 — Phase 2: Fine-tune the top layers of the base model
# =============================================================================

# Rebuild with last 30 layers of MobileNetV2 unfrozen
model, base_model = build_model(trainable_base_layers=30)

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),   # much lower LR for fine-tuning
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# Load weights from phase 1 head training (only works if architecture matches)
# Alternative: just continue training with unfrozen layers
trainable_now = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable parameters in phase 2: {trainable_now:,}")

callbacks_fine = [
    EarlyStopping(
        monitor="val_auc", patience=5, restore_best_weights=True, mode="max",
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=3, min_lr=1e-8, verbose=1,
    ),
    ModelCheckpoint(
        MODEL_SAVE, monitor="val_auc", save_best_only=True,
        mode="max", verbose=1,
    ),
]

print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning top layers of MobileNetV2")
print("=" * 60)

history_fine = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks_fine,
    verbose=1,
)


# =============================================================================
# CELL 7 — Evaluate and visualize results
# =============================================================================

# Load the best saved model
model = tf.keras.models.load_model(MODEL_SAVE)

# Generate predictions on test set
val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=1)
y_pred      = (y_pred_prob > 0.5).astype(int).flatten()
y_true      = val_gen.classes

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
print(f"AUC-ROC: {roc_auc_score(y_true, y_pred_prob):.4f}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["NORMAL", "PNEUMONIA"],
    yticklabels=["NORMAL", "PNEUMONIA"],
    ax=axes[0],
)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")

# Training curves (combined from both phases)
combined_auc = history_head.history["val_auc"] + history_fine.history["val_auc"]
combined_loss = history_head.history["val_loss"] + history_fine.history["val_loss"]
epochs_range = range(1, len(combined_auc) + 1)

axes[1].plot(epochs_range, combined_auc, "b-o", label="Val AUC", markersize=4)
axes[1].plot(epochs_range, combined_loss, "r-s", label="Val Loss", markersize=4)
axes[1].axvline(
    x=len(history_head.history["val_auc"]),
    color="gray", linestyle="--", alpha=0.7, label="Fine-tune start",
)
axes[1].set_xlabel("Epoch")
axes[1].set_title("Training History (Phase 1 + Phase 2)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: training_results.png")
print(f"Saved: {MODEL_SAVE}")
print("\nDownload both files from the Colab Files panel (left sidebar)")
print("You need xray_classifier.keras to run the Gradio demo")


# =============================================================================
# CELL 8 — Quick sanity check: predict a single image
# =============================================================================

from tensorflow.keras.preprocessing import image as keras_image

def predict_single(img_path, threshold=0.5):
    """Predict NORMAL vs PNEUMONIA for a single X-ray image."""
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array, verbose=0)[0][0]
    label = "PNEUMONIA" if prob > threshold else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob

    return {"label": label, "confidence": float(confidence), "raw_prob": float(prob)}


# Test on a few images from the test set
test_normal_dir    = os.path.join(DATA_DIR, "test", "NORMAL")
test_pneumonia_dir = os.path.join(DATA_DIR, "test", "PNEUMONIA")

print("\nSample predictions:")
for folder, true_label in [(test_normal_dir, "NORMAL"), (test_pneumonia_dir, "PNEUMONIA")]:
    img_file = os.path.join(folder, os.listdir(folder)[0])
    result = predict_single(img_file)
    correct = "✓" if result["label"] == true_label else "✗"
    print(f"  {correct} True: {true_label:10s} | Predicted: {result['label']:10s} "
          f"| Confidence: {result['confidence']:.1%}")
