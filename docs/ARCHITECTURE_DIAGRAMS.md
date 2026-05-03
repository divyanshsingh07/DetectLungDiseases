# System architecture & ML diagrams

**Late fusion:** two **independently trained** experts (chest X-ray + EHR/tabular); `fuse_patient_risk` in `app.py` blends two 0–100 scores with `FUSION_WEIGHTS` — **not** a third neural network trained on both modalities.

**Viewing:** [Mermaid](https://mermaid.js.org/) — GitHub / VS Code preview / [mermaid.live](https://mermaid.live).

**Artifacts:** `models/multi_disease_model.pth`, `models/lung_cancer_model.pkl` (+ `*_meta.json`). **Train:** `src/train_multi_disease.py`, `src/train_lung_cancer.py`.

---

## 1. System architecture

High-level ML system: **inputs → modality experts → scores → fusion → patient-level output**.

```mermaid
flowchart TB
  subgraph in["Inputs"]
    I1["Chest X-ray image"]
    I2["EHR / tabular row"]
  end

  subgraph experts["Expert models — trained separately"]
    E1["X-ray expert: ResNet18 multi-class<br/>multi_disease_model.pth"]
    E2["EHR expert: sklearn Pipeline + RF<br/>lung_cancer_model.pkl"]
  end

  subgraph signals["Scores for fusion"]
    S1["s_x: abnormal_probability 0–100"]
    S2["s_e: lung cancer risk 0–100"]
  end

  subgraph fuse["Late fusion — rule-based"]
    F["fuse_patient_risk<br/>normalize weights → weighted avg on 0–1 → clip → risk_band"]
  end

  O["Output: combined score %, tier,<br/>per-modality breakdown"]

  I1 --> E1 --> S1 --> F
  I2 --> E2 --> S2 --> F
  F --> O
```

---

## 2. Model structure

```mermaid
flowchart TB
  subgraph expertA["Chest X-ray — PyTorch ResNet18"]
    XIN["RGB → resize 224, ImageNet norm"]
    XR18["Backbone → GAP 512-d → Linear → C classes"]
    XSM["Softmax"]
    XIN --> XR18 --> XSM
  end

  subgraph expertB["EHR — sklearn Pipeline"]
    TIN["One row: numeric + categorical"]
    CT["ColumnTransformer → RandomForest"]
    PR["predict_proba"]
    TIN --> CT --> PR
  end

  subgraph out["Fusion — fixed weights"]
    AB["s_x: abnormal % from softmax"]
    SE["s_e: cancer % × 100"]
    F["c = clip(w_x·s_x/100 + w_e·s_e/100) → score % + risk_band"]
    XSM --> AB
    PR --> SE
    AB --> F
    SE --> F
  end
```

If `multi_disease_model.pth` is missing, legacy 2-class `model.pth` may be used; `s_x` becomes 1 − P(Normal).

---

## 3. Workflow — training to working output (vertical)

```mermaid
flowchart TB
  DX["① Data: labeled X-ray folders under datasets/"]
  DT["① Data: lung_cancer_dataset.csv"]
  TX["② Train: src/train_multi_disease.py"]
  TT["② Train: src/train_lung_cancer.py"]
  MX["③ Save: models/multi_disease_model.pth + multi_disease_meta.json"]
  MT["③ Save: models/lung_cancer_model.pkl + lung_cancer_model_meta.json"]
  LD["④ Load: restore ResNet18 + sklearn Pipeline into memory"]
  IM["⑤ Input: chest X-ray image"]
  ER["⑤ Input: EHR row matching meta columns"]
  SX["⑥ X-ray: preprocess 224 → ResNet18 → softmax → s_x 0–100"]
  SE["⑥ EHR: transform row → RandomForest → s_e 0–100"]
  FU["⑦ fuse_patient_risk: normalize FUSION_WEIGHTS, weighted sum on 0–1, clip, risk_band"]
  OUT["⑧ Working output: combined score %, tier, X-ray + EHR breakdown"]

  DX --> TX --> MX --> LD
  DT --> TT --> MT --> LD
  LD --> IM --> SX --> FU
  LD --> ER --> SE --> FU
  FU --> OUT
```

No training step optimizes the combined score. Missing EHR → **X-ray only** (EHR weight 0).

---

## 4. UML — class diagram

```mermaid
classDiagram
  class ChestXRayExpert {
    +ResNet18 backbone
    +Linear 512→C
    +forward(image Tensor)
    +softmax logits
  }

  class EHRExpert {
    ColumnTransformer
    RandomForestClassifier
    +predict_proba(row)
  }

  class XrayResult {
    <<dict>>
    abnormal_probability
    class_probabilities
    prediction
  }

  class LungResult {
    <<dict>>
    probability
  }

  class FusedPatientScore {
    <<dict>>
    score
    risk_band
    method
    ehr_used
    weights
    components
  }

  class FusionPolicy {
    <<configuration>>
    FUSION_WEIGHTS
  }

  class RiskFusion {
    <<app.py>>
    +fuse_patient_risk(xray_result, lung_result) dict
  }

  ChestXRayExpert ..> XrayResult : produces
  EHRExpert ..> LungResult : produces
  FusionPolicy ..> RiskFusion : weights

  note for ChestXRayExpert "torchvision; artifact .pth"
  note for EHRExpert "sklearn Pipeline; artifact .pkl"
  note for FusedPatientScore "keys: score, risk_band, method, …"

  XrayResult --> RiskFusion : input
  LungResult --> RiskFusion : input
  RiskFusion ..> FusedPatientScore : returns
```

---

## 5. UML — sequence diagram (combined inference)

```mermaid
sequenceDiagram
  autonumber
  participant Img as Chest image
  participant X as ResNet18 + preprocess
  participant Row as EHR row
  participant R as sklearn Pipeline
  participant F as fuse_patient_risk
  participant Out as Fused score dict

  Img->>X: tensor 1×3×224×224
  X->>X: softmax
  X->>F: xray_result
  Row->>R: DataFrame 1 row
  R->>F: lung_result
  F->>F: normalize weights, clip, risk_band
  F->>Out: score, risk_band, components
```

---

More detail: [pipline.md](pipline.md) · [README.md](../README.md)
