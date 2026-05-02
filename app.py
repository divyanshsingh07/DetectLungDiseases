from io import BytesIO
from pathlib import Path
import base64
import json
import os
import pickle
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
EVAL_RESULTS_DIR = BASE_DIR / "evaluation_results"

MODEL_PATH = MODELS_DIR / "model.pth"
MULTI_MODEL_PATH = MODELS_DIR / "multi_disease_model.pth"
MULTI_META_PATH = MODELS_DIR / "multi_disease_meta.json"
LUNG_MODEL_PATH = MODELS_DIR / "lung_cancer_model.pkl"
LUNG_META_PATH = MODELS_DIR / "lung_cancer_model_meta.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Legacy 2-class pneumonia model
CLASS_NAMES = ["Normal", "Pneumonia"]

# Multi-disease ResNet18 (5 classes by default; gets overridden by meta)
DEFAULT_MULTI_CLASSES = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Lung_Opacity"]

# Diseases the combined endpoint treats as "abnormal" when computing
# the fused respiratory health score.
ABNORMAL_CLASSES = {"Pneumonia", "COVID-19", "Tuberculosis", "Lung_Opacity"}
FUSION_WEIGHTS = {
    "xray_abnormal": 0.5,
    "ehr_lung_cancer": 0.5,
}

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Pneumonia model (ResNet18, legacy 2-class)
# ---------------------------------------------------------------------------
def build_pneumonia_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


model = build_pneumonia_model()
transform = get_transform()


# ---------------------------------------------------------------------------
# Multi-disease ResNet18 (5-class: Normal/Pneumonia/COVID-19/TB/Lung_Opacity)
# ---------------------------------------------------------------------------
def load_multi_disease_model():
    """Return (model, meta) or (None, None) if the artefacts aren't present."""
    if not MULTI_MODEL_PATH.exists():
        return None, None

    meta = None
    if MULTI_META_PATH.exists():
        with open(MULTI_META_PATH, "r") as f:
            meta = json.load(f)
    class_names = (meta or {}).get("class_names") or DEFAULT_MULTI_CLASSES

    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(class_names))
    try:
        state_dict = torch.load(MULTI_MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MULTI_MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net, meta


multi_model, multi_meta = load_multi_disease_model()
multi_classes = (multi_meta or {}).get("class_names") or DEFAULT_MULTI_CLASSES


# ---------------------------------------------------------------------------
# Lung cancer risk model (sklearn pipeline trained on EHR data)
# ---------------------------------------------------------------------------
def load_lung_cancer_artifacts():
    if not LUNG_MODEL_PATH.exists() or not LUNG_META_PATH.exists():
        return None, None
    with open(LUNG_MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    with open(LUNG_META_PATH, "r") as f:
        meta = json.load(f)
    return pipe, meta


lung_model, lung_meta = load_lung_cancer_artifacts()


def risk_band(probability: float) -> dict:
    """Map a probability into a human-readable risk tier."""
    pct = probability * 100.0
    if pct < 25:
        return {"label": "Low Risk", "tier": "low", "range": "<25%"}
    if pct < 55:
        return {"label": "Moderate Risk", "tier": "moderate", "range": "25–55%"}
    if pct < 80:
        return {"label": "High Risk", "tier": "high", "range": "55–80%"}
    return {"label": "Very High Risk", "tier": "critical", "range": ">=80%"}


def _probability_from_percent(value, field_name: str) -> float:
    """Convert a 0-100 model output into a bounded 0-1 probability."""
    try:
        probability = float(value) / 100.0
    except (TypeError, ValueError):
        raise ValueError(f"Invalid probability for {field_name}.")
    return float(np.clip(probability, 0.0, 1.0))


def fuse_patient_risk(xray_result: dict, lung_result: Optional[dict]) -> dict:
    """Fuse X-ray abnormality and optional EHR cancer risk into one patient score.

    When ``lung_result`` is None or missing probability, the score uses only the
    X-ray abnormality signal (same weighting semantics as an all-X-ray fusion).
    """
    if "abnormal_probability" not in xray_result:
        raise ValueError("X-ray result is missing abnormal_probability.")

    abnormal_prob = _probability_from_percent(
        xray_result["abnormal_probability"], "xray.abnormal_probability"
    )

    if lung_result is None or "probability" not in lung_result:
        overall = float(np.clip(abnormal_prob, 0.0, 1.0))
        return {
            "score": round(overall * 100.0, 2),
            "risk_band": risk_band(overall),
            "method": "xray_only",
            "ehr_used": False,
            "weights": {"xray_abnormal": 1.0, "ehr_lung_cancer": 0.0},
            "components": {
                "xray_abnormal": round(abnormal_prob * 100.0, 2),
                "ehr_lung_cancer": None,
            },
        }

    cancer_prob = _probability_from_percent(
        lung_result["probability"], "lung_cancer.probability"
    )

    total_weight = sum(FUSION_WEIGHTS.values())
    if total_weight <= 0:
        raise ValueError("Fusion weights must sum to a positive value.")

    xray_weight = FUSION_WEIGHTS["xray_abnormal"] / total_weight
    ehr_weight = FUSION_WEIGHTS["ehr_lung_cancer"] / total_weight
    overall = (xray_weight * abnormal_prob) + (ehr_weight * cancer_prob)
    overall = float(np.clip(overall, 0.0, 1.0))

    return {
        "score": round(overall * 100.0, 2),
        "risk_band": risk_band(overall),
        "method": "weighted_average",
        "ehr_used": True,
        "weights": {
            "xray_abnormal": round(xray_weight, 3),
            "ehr_lung_cancer": round(ehr_weight, 3),
        },
        "components": {
            "xray_abnormal": round(abnormal_prob * 100.0, 2),
            "ehr_lung_cancer": round(cancer_prob * 100.0, 2),
        },
    }


def lung_cancer_recommendations(tier: str, payload: dict) -> list:
    recs = []
    if payload.get("copd_diagnosis") == "Yes":
        recs.append("Known COPD history. Maintain regular pulmonology follow-ups.")
    try:
        pack_years = float(payload.get("pack_years", 0) or 0)
    except (TypeError, ValueError):
        pack_years = 0.0
    if pack_years >= 20:
        recs.append(
            f"Pack-year count of {pack_years:.0f} meets USPSTF LDCT screening thresholds."
        )
    if payload.get("radon_exposure") == "High":
        recs.append("High radon exposure reported. Consider in-home radon mitigation.")
    if payload.get("asbestos_exposure") == "Yes":
        recs.append("Prior asbestos exposure. Share this with your physician for monitoring.")
    if payload.get("family_history") == "Yes":
        recs.append("Family history noted. Discuss genetic risk and screening cadence.")

    if tier in {"high", "critical"}:
        recs.insert(0, "Seek evaluation by a pulmonologist for individualized screening.")
    elif tier == "moderate":
        recs.insert(0, "Discuss low-dose CT screening with your primary care physician.")
    else:
        recs.insert(0, "Continue annual wellness visits and avoid new tobacco exposure.")
    return recs


def disease_recommendations(top_class: str, top_prob: float) -> list:
    recs: list = []
    if top_class == "Normal":
        recs.append("X-ray appears normal. Continue routine wellness check-ups.")
        return recs
    if top_class == "Pneumonia":
        recs.append("Findings suggest pneumonia. Please consult a clinician for sputum/blood work.")
        recs.append("Empirical antibiotic therapy may be considered after physician review.")
    elif top_class == "COVID-19":
        recs.append("COVID-19 pattern suspected. Confirm with RT-PCR or antigen testing.")
        recs.append("Isolate, monitor SpO2, and seek care if breathing worsens.")
    elif top_class == "Tuberculosis":
        recs.append("TB pattern suspected. Sputum smear / GeneXpert testing strongly recommended.")
        recs.append("Notify public-health authorities and screen close contacts.")
    elif top_class == "Lung_Opacity":
        recs.append("Non-specific lung opacity detected. Clinical correlation and follow-up imaging advised.")
        recs.append("Consider HRCT to characterise the opacity further.")

    if top_prob < 70:
        recs.append("Confidence is moderate; a radiologist read is strongly recommended.")
    return recs


# ---------------------------------------------------------------------------
# Grad-CAM visualisation for the multi-disease model (falls back to legacy)
# ---------------------------------------------------------------------------
def generate_gradcam_overlay(image: Image.Image, target_model: nn.Module, target_idx: int = None) -> str:
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    tensor = transform(img_resized).unsqueeze(0).to(device)

    activations = []
    gradients = []

    def forward_hook(_module, _inp, output):
        activations.append(output)

    def backward_hook(_module, grad_input, grad_output):
        gradients.append(grad_output[0])

    layer = target_model.layer4[-1]
    hook_f = layer.register_forward_hook(forward_hook)
    hook_b = layer.register_full_backward_hook(backward_hook)

    logits = target_model(tensor)
    if target_idx is None:
        target_idx = int(torch.argmax(logits, dim=1).item())
    score = logits[:, target_idx]

    target_model.zero_grad()
    score.backward()

    hook_f.remove()
    hook_b.remove()

    grad = gradients[0].detach()
    act = activations[0].detach()
    pooled_grad = torch.mean(grad, dim=(0, 2, 3))
    cam = torch.sum(pooled_grad[None, :, None, None] * act, dim=1).squeeze()
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    cam_np = cam.cpu().numpy()
    heatmap = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    heatmap_np = np.array(heatmap).astype(np.float32) / 255.0

    overlay = img_array.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + (heatmap_np * 0.75), 0, 1)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1 - 0.45 * heatmap_np), 0, 1)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1 - 0.45 * heatmap_np), 0, 1)

    out_img = Image.fromarray((overlay * 255).astype(np.uint8))
    buff = BytesIO()
    out_img.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")


def run_multi_disease_inference(image: Image.Image) -> dict:
    """Multi-class chest X-ray prediction. Returns probs + top class + heatmap."""
    if multi_model is None:
        # Fall back to legacy 2-class model so the app still works.
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        idx = int(np.argmax(probs))
        cls = CLASS_NAMES[idx]
        return {
            "model": "pneumonia_legacy",
            "prediction": cls,
            "confidence": round(probs[idx] * 100.0, 2),
            "probabilities": {CLASS_NAMES[i]: round(probs[i] * 100.0, 2) for i in range(len(CLASS_NAMES))},
            "abnormal_probability": round((1.0 - probs[0]) * 100.0, 2),
            "heatmap_overlay": generate_gradcam_overlay(image, model, target_idx=idx),
        }

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = multi_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    idx = int(np.argmax(probs))
    cls = multi_classes[idx]

    abnormal_prob = sum(
        probs[i] for i, name in enumerate(multi_classes) if name in ABNORMAL_CLASSES
    )

    return {
        "model": "multi_disease",
        "prediction": cls,
        "confidence": round(probs[idx] * 100.0, 2),
        "probabilities": {multi_classes[i]: round(probs[i] * 100.0, 2) for i in range(len(multi_classes))},
        "abnormal_probability": round(abnormal_prob * 100.0, 2),
        "heatmap_overlay": generate_gradcam_overlay(image, multi_model, target_idx=idx),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _nav_context(nav_active: str) -> dict:
    """Shared template context for the site shell."""
    return {
        "nav_active": nav_active,
        "has_lung": lung_model is not None,
        "has_multi": multi_model is not None,
    }


def _dashboard_schemas():
    form_schema = None
    if lung_meta is not None:
        form_schema = {
            "numeric_features": lung_meta["numeric_features"],
            "categorical_features": lung_meta["categorical_features"],
            "feature_value_options": lung_meta["feature_value_options"],
            "metrics": lung_meta["metrics"],
        }
    multi_schema = None
    if multi_meta is not None:
        multi_schema = {
            "class_names": multi_meta.get("class_names", DEFAULT_MULTI_CLASSES),
            "metrics": multi_meta.get("metrics", {}),
            "per_class_counts": multi_meta.get("per_class_counts", {}),
            "n_train": multi_meta.get("n_train"),
            "n_val": multi_meta.get("n_val"),
        }
    return form_schema, multi_schema


def _enrich_accuracy_report(report: dict) -> dict:
    """Add deployment fusion metadata for the metrics page."""
    xray_metrics = report.get("multi_disease_model") or {}
    ehr_metrics = report.get("lung_cancer_model") or {}
    total_weight = sum(FUSION_WEIGHTS.values()) or 1.0
    xray_w_cfg = FUSION_WEIGHTS["xray_abnormal"] / total_weight
    ehr_w_cfg = FUSION_WEIGHTS["ehr_lung_cancer"] / total_weight

    xray_accuracy = xray_metrics.get("accuracy")
    ehr_accuracy = ehr_metrics.get("accuracy")
    fusion_score = None
    ehr_used_in_proxy = False
    display_xray_w, display_ehr_w = xray_w_cfg, ehr_w_cfg

    if xray_accuracy is not None and ehr_accuracy is not None:
        fusion_score = (xray_w_cfg * float(xray_accuracy)) + (ehr_w_cfg * float(ehr_accuracy))
        ehr_used_in_proxy = True
    elif xray_accuracy is not None:
        fusion_score = float(xray_accuracy)
        display_xray_w, display_ehr_w = 1.0, 0.0
    elif ehr_accuracy is not None:
        fusion_score = float(ehr_accuracy)
        display_xray_w, display_ehr_w = 0.0, 1.0
        ehr_used_in_proxy = True

    report["fusion_metrics"] = {
        "method": "weighted_average",
        "xray_weight": display_xray_w,
        "ehr_weight": display_ehr_w,
        "xray_accuracy": xray_accuracy,
        "ehr_accuracy": ehr_accuracy,
        "score_proxy": fusion_score,
        "ehr_used_in_proxy": ehr_used_in_proxy,
        "has_patient_level_accuracy": False,
        "note": (
            "Fusion combines the X-ray abnormality probability and optional EHR lung cancer "
            "probability at prediction time. The displayed fusion score is a model-accuracy "
            "proxy (weighted when both models were evaluated, otherwise the available modality). "
            "There is no single paired patient-level test set for a true multimodal accuracy."
        ),
    }
    return report


@app.route("/", methods=["GET"])
def index():
    """Marketing / landing page."""
    return render_template("landing.html", **_nav_context("home"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Main analysis UI (X-ray, EHR, combined)."""
    lung_schema, multi_schema = _dashboard_schemas()
    return render_template(
        "dashboard.html",
        lung_schema=lung_schema,
        multi_schema=multi_schema,
        **_nav_context("dashboard"),
    )


@app.route("/how-it-works", methods=["GET"])
def how_it_works():
    """Static explanation of models and limitations."""
    return render_template("how_it_works.html", **_nav_context("how"))


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No image selected."}), 400

    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, or JPEG."}), 400

    try:
        image = Image.open(BytesIO(file.read())).convert("RGB")
        xray = run_multi_disease_inference(image)
        recs = disease_recommendations(xray["prediction"], xray["confidence"])

        return jsonify(
            {
                "model": xray["model"],
                "prediction": xray["prediction"],
                "confidence": xray["confidence"],
                "probabilities": xray["probabilities"],
                "abnormal_probability": xray["abnormal_probability"],
                "recommendations": recs,
                "analysis": {
                    "heatmap_overlay": xray["heatmap_overlay"],
                    "steps": [
                        "Image resized to 224x224 and normalized with ImageNet statistics.",
                        "ResNet18 trained on 6,700 chest X-rays from three public datasets.",
                        f"Probabilities computed across {len(xray['probabilities'])} disease classes.",
                        "Grad-CAM highlights the regions that most influenced this prediction.",
                    ],
                },
            }
        )
    except Exception as exc:
        app.logger.exception("Predict failed: %s", exc)
        return jsonify({"error": "Failed to process this image. Please try another file."}), 500


def _lung_payload_from_request(req) -> dict:
    if req.is_json:
        return req.get_json(silent=True) or {}
    return {k: v for k, v in req.form.items()}


def _lung_ehr_payload_complete(payload: dict) -> bool:
    """True if every EHR field required by metadata is present and non-empty."""
    if lung_meta is None:
        return False
    numeric = lung_meta["numeric_features"]
    categorical = lung_meta["categorical_features"]
    for col in numeric + categorical:
        v = payload.get(col)
        if v is None:
            return False
        if isinstance(v, str) and v.strip() == "":
            return False
    return True


def _run_lung_cancer_model(payload: dict) -> dict:
    if lung_model is None or lung_meta is None:
        raise RuntimeError(
            "Lung cancer model not found. Run `python3 src/train_lung_cancer.py` first."
        )

    numeric = lung_meta["numeric_features"]
    categorical = lung_meta["categorical_features"]
    options = lung_meta["feature_value_options"]

    missing = [c for c in numeric + categorical if c not in payload or payload[c] in ("", None)]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    row = {}
    for col in numeric:
        try:
            row[col] = float(payload[col])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{col}' must be numeric.")
    for col in categorical:
        value = str(payload[col])
        allowed = options.get(col, [])
        if allowed and value not in allowed:
            raise ValueError(
                f"Field '{col}' must be one of {allowed}; got '{value}'."
            )
        row[col] = value

    df = pd.DataFrame([row])
    probs = lung_model.predict_proba(df)[0]
    prob_cancer = float(probs[1])
    prob_no = float(probs[0])
    band = risk_band(prob_cancer)
    recs = lung_cancer_recommendations(band["tier"], row)

    return {
        "probability": round(prob_cancer * 100.0, 2),
        "probability_no_cancer": round(prob_no * 100.0, 2),
        "risk_band": band,
        "recommendations": recs,
        "inputs": row,
    }


@app.route("/predict-lung-cancer", methods=["POST"])
def predict_lung_cancer():
    try:
        payload = _lung_payload_from_request(request)
        result = _run_lung_cancer_model(payload)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re_:
        return jsonify({"error": str(re_)}), 500
    except Exception:
        return jsonify({"error": "Failed to score the EHR record."}), 500


@app.route("/predict-combined", methods=["POST"])
def predict_combined():
    """Run multi-disease X-ray analysis and optionally EHR cancer-risk, then fuse.

    Expects multipart/form-data with:
      - image: chest X-ray file (required)
      - EHR fields as flat form entries (optional). When all required fields are
        present and the EHR model is loaded, fusion uses both modalities; otherwise
        the combined score uses the X-ray abnormality signal only.
    """
    if "image" not in request.files:
        return jsonify({"error": "Chest X-ray image is required."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No image selected."}), 400

    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported image type. Use PNG, JPG, or JPEG."}), 400

    try:
        image = Image.open(BytesIO(file.read())).convert("RGB")
        xray = run_multi_disease_inference(image)
    except Exception:
        return jsonify({"error": "Failed to analyse the chest X-ray."}), 500

    ehr_payload = {k: v for k, v in request.form.items()}
    lung_result = None
    if _lung_ehr_payload_complete(ehr_payload):
        try:
            lung_result = _run_lung_cancer_model(ehr_payload)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except RuntimeError as re_:
            return jsonify({"error": str(re_)}), 500
        except Exception:
            return jsonify({"error": "Failed to score the EHR inputs."}), 500

    try:
        fusion = fuse_patient_risk(xray, lung_result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 500

    summary_lines = [
        f"X-ray top finding: {xray['prediction']} ({xray['confidence']}% confidence).",
        f"Abnormal X-ray signal (any disease): {xray['abnormal_probability']}%.",
    ]
    if lung_result is not None:
        summary_lines.append(
            f"EHR-derived lung cancer risk: {lung_result['probability']}% "
            f"({lung_result['risk_band']['label']})."
        )
    else:
        summary_lines.append(
            "EHR not applied (fields omitted, incomplete, or model unavailable); "
            "score reflects chest X-ray only."
        )
    summary_lines.append(
        f"Combined respiratory health score: {fusion['score']}% "
        f"({fusion['risk_band']['label']})."
    )

    response_body = {
        "xray": {
            "prediction": xray["prediction"],
            "confidence": xray["confidence"],
            "probabilities": xray["probabilities"],
            "abnormal_probability": xray["abnormal_probability"],
            "model": xray["model"],
            "heatmap_overlay": xray["heatmap_overlay"],
            "recommendations": disease_recommendations(xray["prediction"], xray["confidence"]),
        },
        "lung_cancer": lung_result,
        "combined": {
            "score": fusion["score"],
            "risk_band": fusion["risk_band"],
            "method": fusion["method"],
            "ehr_used": fusion.get("ehr_used", False),
            "weights": fusion["weights"],
            "components": fusion["components"],
            "summary": summary_lines,
        },
    }
    return jsonify(response_body)


@app.route("/model-accuracy", methods=["GET"])
def model_accuracy():
    """Display model accuracy metrics and visualizations."""
    report_path = EVAL_RESULTS_DIR / "evaluation_report.json"

    ctx = _nav_context("metrics")
    if not report_path.exists():
        return render_template(
            "accuracy.html",
            evaluated=False,
            error="Evaluation not yet run. Click 'Run Evaluation' to generate metrics.",
            **ctx,
        )

    try:
        with open(report_path, "r") as f:
            report = json.load(f)
        report = _enrich_accuracy_report(report)
        return render_template("accuracy.html", evaluated=True, report=report, **ctx)
    except Exception as e:
        return render_template(
            "accuracy.html",
            evaluated=False,
            error=f"Error loading results: {str(e)}",
            **ctx,
        )


@app.route("/run-evaluation", methods=["POST"])
def run_evaluation():
    """Trigger the comprehensive evaluation script and stream back results."""
    import subprocess

    try:
        result = subprocess.run(
            ["python", str(BASE_DIR / "src" / "comprehensive_evaluation.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode == 0:
            report_path = EVAL_RESULTS_DIR / "evaluation_report.json"
            with open(report_path, "r") as f:
                report = json.load(f)
            return jsonify(
                {
                    "success": True,
                    "message": "Evaluation completed successfully!",
                    "report": report,
                }
            )
        return jsonify({"success": False, "error": f"Evaluation failed: {result.stderr}"}), 500

    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Evaluation timed out."}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Error running evaluation: {str(e)}"}), 500


@app.route("/evaluation-image/<path:filename>")
def evaluation_image(filename: str):
    """Serve evaluation result images so the dashboard can embed them."""
    return send_from_directory(str(EVAL_RESULTS_DIR), filename)


if __name__ == "__main__":
    # PORT matches docker-compose host mapping when set (e.g. PORT=8080).
    # FLASK_RUN_HOST=0.0.0.0 allows other devices on your LAN to reach the dev server.
    _port = int(os.environ.get("PORT", "8080"))
    _host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    app.run(debug=True, host=_host, port=_port)
