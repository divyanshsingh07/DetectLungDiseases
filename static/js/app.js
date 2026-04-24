/* ---------------------------------------------------------------------------
 * Tab switching
 * ------------------------------------------------------------------------- */
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    tabs.forEach((t) => t.classList.toggle("active", t === tab));
    panels.forEach((p) =>
      p.classList.toggle("active", p.id === `tab-${target}`)
    );
  });
});

/* ---------------------------------------------------------------------------
 * Shared helpers
 * ------------------------------------------------------------------------- */
const supportedTypes = ["image/jpeg", "image/png"];

const DISEASE_META = {
  Normal: { tier: "low", icon: "fa-circle-check", color: "var(--success)" },
  Pneumonia: { tier: "high", icon: "fa-lungs-virus", color: "var(--error)" },
  "COVID-19": { tier: "high", icon: "fa-virus-covid", color: "var(--error)" },
  Tuberculosis: { tier: "high", icon: "fa-bacterium", color: "var(--error)" },
  Lung_Opacity: { tier: "moderate", icon: "fa-cloud", color: "var(--warning)" },
};

function diseaseMeta(name) {
  return (
    DISEASE_META[name] || {
      tier: "moderate",
      icon: "fa-circle-info",
      color: "var(--accent)",
    }
  );
}

function setMessage(el, text, isError = false) {
  if (!el) return;
  el.textContent = text;
  el.style.color = isError ? "var(--error, #EF4444)" : "var(--muted, #64748B)";
}

function validateImage(file) {
  if (!file) return "Please select an image file.";
  if (!supportedTypes.includes(file.type))
    return "Unsupported file type. Please upload JPG or PNG.";
  return null;
}

function renderProbGrid(container, probabilities, predicted) {
  container.innerHTML = "";
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  entries.forEach(([name, prob]) => {
    const meta = diseaseMeta(name);
    const item = document.createElement("div");
    item.className = "prob-item";
    if (name === predicted) item.classList.add("prob-top");
    item.innerHTML = `
      <div class="prob-row">
        <span><i class="fa-solid ${meta.icon}" style="color:${meta.color}"></i> ${name.replace("_", " ")}</span>
        <strong>${prob}%</strong>
      </div>
      <div class="prob-bar"><span style="width:${Math.min(100, prob)}%; background:${meta.color}"></span></div>
    `;
    container.appendChild(item);
  });
}

/* ---------------------------------------------------------------------------
 * X-RAY TAB
 * ------------------------------------------------------------------------- */
const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const previewWrapper = document.getElementById("previewWrapper");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const message = document.getElementById("message");

const resultPanel = document.getElementById("resultPanel");
const resultTitle = document.getElementById("resultTitle");
const resultSubtitle = document.getElementById("resultSubtitle");
const resultIcon = document.getElementById("resultIcon");
const probGrid = document.getElementById("probGrid");
const xrayRecs = document.getElementById("xrayRecs");
const analysisHeatmap = document.getElementById("analysisHeatmap");
const analysisSteps = document.getElementById("analysisSteps");
const confidenceFill = document.getElementById("confidenceFill");

function resetXrayResult() {
  resultPanel.classList.add("hidden");
  resultPanel.classList.remove("result-normal", "result-abnormal");
  resultIcon.className = "fa-solid fa-circle-info";
  if (analysisHeatmap) analysisHeatmap.src = "";
  if (analysisSteps) analysisSteps.innerHTML = "";
  if (probGrid) probGrid.innerHTML = "";
  if (xrayRecs) xrayRecs.innerHTML = "";
  if (confidenceFill) confidenceFill.style.width = "0%";
}

function showXrayPreview(file) {
  previewImage.src = URL.createObjectURL(file);
  previewWrapper.classList.remove("hidden");
}

function handleXraySelectedFile(file) {
  const error = validateImage(file);
  if (error) {
    setMessage(message, error, true);
    predictBtn.disabled = true;
    previewWrapper.classList.add("hidden");
    resetXrayResult();
    return;
  }
  setMessage(message, "Image ready. Click Analyze Image to run prediction.");
  showXrayPreview(file);
  predictBtn.disabled = false;
  resetXrayResult();
}

if (fileInput) {
  fileInput.addEventListener("change", (event) =>
    handleXraySelectedFile(event.target.files[0])
  );
}

if (dropZone) {
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () =>
    dropZone.classList.remove("dragover")
  );
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    fileInput.files = event.dataTransfer.files;
    handleXraySelectedFile(file);
  });
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    const error = validateImage(file);
    if (error) return setMessage(message, error, true);

    const formData = new FormData();
    formData.append("image", file);
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing image...';
    setMessage(message, "Running AI screening...");

    try {
      const response = await fetch("/predict", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed.");

      const meta = diseaseMeta(data.prediction);
      const isNormal = data.prediction === "Normal";

      resultPanel.classList.remove("hidden");
      resultPanel.classList.toggle("result-normal", isNormal);
      resultPanel.classList.toggle("result-abnormal", !isNormal);

      resultIcon.className = `fa-solid ${meta.icon}`;
      resultIcon.style.color = meta.color;
      resultTitle.textContent = data.prediction.replace("_", " ");
      resultSubtitle.textContent =
        `Confidence ${data.confidence}%`;
      if (confidenceFill) confidenceFill.style.width = `${Math.min(100, data.confidence)}%`;

      if (probGrid) renderProbGrid(probGrid, data.probabilities, data.prediction);

      if (xrayRecs) {
        xrayRecs.innerHTML = "";
        (data.recommendations || []).slice(0, 2).forEach((rec) => {
          const li = document.createElement("li");
          li.textContent = rec;
          xrayRecs.appendChild(li);
        });
      }

      if (analysisHeatmap && data.analysis && data.analysis.heatmap_overlay) {
        analysisHeatmap.src = data.analysis.heatmap_overlay;
      }
      if (analysisSteps) {
        analysisSteps.innerHTML = "";
        (data.analysis?.steps || []).forEach((stepText, index) => {
          const li = document.createElement("li");
          li.textContent = `${index + 1}. ${stepText}`;
          analysisSteps.appendChild(li);
        });
      }
      setMessage(message, "Analysis complete.");
    } catch (err) {
      setMessage(message, err.message || "Something went wrong during prediction.", true);
      resetXrayResult();
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Image';
    }
  });
}

/* ---------------------------------------------------------------------------
 * EHR TAB (lung cancer risk)
 * ------------------------------------------------------------------------- */
const ehrForm = document.getElementById("ehrForm");
const ehrMessage = document.getElementById("ehrMessage");
const ehrResult = document.getElementById("ehrResult");
const ehrResultTitle = document.getElementById("ehrResultTitle");
const ehrResultSubtitle = document.getElementById("ehrResultSubtitle");
const ehrRiskFill = document.getElementById("ehrRiskFill");
const ehrRecs = document.getElementById("ehrRecs");

function applyRiskBand(panel, tier) {
  panel.classList.remove("risk-low", "risk-moderate", "risk-high", "risk-critical");
  panel.classList.add(`risk-${tier}`);
}

function renderRiskResult(data) {
  ehrResult.classList.remove("hidden");
  applyRiskBand(ehrResult, data.risk_band.tier);
  ehrResultTitle.textContent = `${data.risk_band.label} · ${data.probability}%`;
  ehrResultSubtitle.textContent = `Estimated lung cancer probability (range ${data.risk_band.range}).`;
  ehrRiskFill.style.width = `${Math.min(100, Math.max(3, data.probability))}%`;
  ehrRecs.innerHTML = "";
  data.recommendations.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    ehrRecs.appendChild(li);
  });
}

if (ehrForm) {
  ehrForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(ehrForm).entries());
    setMessage(ehrMessage, "Scoring EHR record...");

    try {
      const response = await fetch("/predict-lung-cancer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to score EHR record.");
      renderRiskResult(data);
      setMessage(ehrMessage, "Risk scoring complete.");
    } catch (err) {
      setMessage(ehrMessage, err.message || "Something went wrong.", true);
      ehrResult.classList.add("hidden");
    }
  });
}

/* ---------------------------------------------------------------------------
 * COMBINED TAB
 * ------------------------------------------------------------------------- */
const combinedForm = document.getElementById("combinedForm");
const combinedFileInput = document.getElementById("combinedFileInput");
const combinedDropZone = document.getElementById("combinedDropZone");
const combinedPreviewWrapper = document.getElementById("combinedPreviewWrapper");
const combinedPreviewImage = document.getElementById("combinedPreviewImage");
const combinedBtn = document.getElementById("combinedBtn");
const combinedMessage = document.getElementById("combinedMessage");
const combinedResult = document.getElementById("combinedResult");
const combinedTitle = document.getElementById("combinedTitle");
const combinedSubtitle = document.getElementById("combinedSubtitle");
const combinedXrayEl = document.getElementById("combinedXray");
const combinedXrayDetail = document.getElementById("combinedXrayDetail");
const combinedEhrEl = document.getElementById("combinedEhr");
const combinedOverallEl = document.getElementById("combinedOverall");
const combinedSummaryList = document.getElementById("combinedSummary");
const combinedProbGrid = document.getElementById("combinedProbGrid");

function handleCombinedFile(file) {
  const error = validateImage(file);
  if (error) {
    setMessage(combinedMessage, error, true);
    combinedBtn.disabled = true;
    combinedPreviewWrapper.classList.add("hidden");
    return;
  }
  combinedPreviewImage.src = URL.createObjectURL(file);
  combinedPreviewWrapper.classList.remove("hidden");
  combinedBtn.disabled = false;
  setMessage(combinedMessage, "Image ready. Fill EHR fields and submit.");
}

if (combinedFileInput) {
  combinedFileInput.addEventListener("change", (event) =>
    handleCombinedFile(event.target.files[0])
  );
}

if (combinedDropZone) {
  combinedDropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    combinedDropZone.classList.add("dragover");
  });
  combinedDropZone.addEventListener("dragleave", () =>
    combinedDropZone.classList.remove("dragover")
  );
  combinedDropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    combinedDropZone.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    combinedFileInput.files = event.dataTransfer.files;
    handleCombinedFile(file);
  });
}

if (combinedForm) {
  combinedForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = combinedFileInput.files[0];
    const error = validateImage(file);
    if (error) return setMessage(combinedMessage, error, true);

    const formData = new FormData(combinedForm);
    combinedBtn.disabled = true;
    combinedBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    setMessage(combinedMessage, "Running combined assessment...");

    try {
      const response = await fetch("/predict-combined", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Combined assessment failed.");

      combinedResult.classList.remove("hidden");
      combinedResult.classList.remove("risk-low", "risk-moderate", "risk-high", "risk-critical");
      combinedResult.classList.add(`risk-${data.combined.risk_band.tier}`);

      combinedTitle.textContent = `${data.combined.risk_band.label} · Score ${data.combined.score}%`;
      combinedSubtitle.textContent =
        "Fused multi-disease X-ray finding + EHR-derived lung cancer risk.";

      combinedXrayEl.textContent =
        `${data.xray.prediction.replace("_", " ")} (${data.xray.confidence}%)`;
      if (combinedXrayDetail)
        combinedXrayDetail.textContent =
          `Abnormal signal: ${data.xray.abnormal_probability}%`;
      combinedEhrEl.textContent =
        `${data.lung_cancer.probability}% · ${data.lung_cancer.risk_band.label}`;
      combinedOverallEl.textContent =
        `${data.combined.score}% · ${data.combined.risk_band.label}`;

      renderProbGrid(combinedProbGrid, data.xray.probabilities, data.xray.prediction);

      combinedSummaryList.innerHTML = "";
      data.combined.summary.forEach((line) => {
        const li = document.createElement("li");
        li.textContent = line;
        combinedSummaryList.appendChild(li);
      });
      (data.xray.recommendations || []).forEach((rec) => {
        const li = document.createElement("li");
        li.textContent = `X-ray rec: ${rec}`;
        combinedSummaryList.appendChild(li);
      });
      data.lung_cancer.recommendations.forEach((rec) => {
        const li = document.createElement("li");
        li.textContent = `EHR rec: ${rec}`;
        combinedSummaryList.appendChild(li);
      });

      setMessage(combinedMessage, "Combined assessment complete.");
    } catch (err) {
      setMessage(combinedMessage, err.message || "Something went wrong.", true);
      combinedResult.classList.add("hidden");
    } finally {
      combinedBtn.disabled = false;
      combinedBtn.innerHTML =
        '<i class="fa-solid fa-layer-group"></i> Run Combined Assessment';
    }
  });
}
