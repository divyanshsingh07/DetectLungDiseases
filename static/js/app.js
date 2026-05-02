/* ---------------------------------------------------------------------------
 * Tab switching
 * ------------------------------------------------------------------------- */
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    tabs.forEach((t) => {
      const on = t === tab;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
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

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function clampPercent(value, min = 0.5, max = 99.5) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

function formatPercent(value) {
  return `${clampPercent(value).toFixed(1)}%`;
}

function normalizedProbabilityView(probabilities) {
  const entries = Object.entries(probabilities || {});
  if (!entries.length) return {};
  const adjusted = entries.map(([label, raw]) => {
    const n = Number(raw);
    const base = Number.isFinite(n) ? n : 0;
    return [label, clampPercent(base, 0.5, 99.0)];
  });
  const sum = adjusted.reduce((acc, [, n]) => acc + n, 0) || 1;
  const scale = 100 / sum;
  const normalized = {};
  adjusted.forEach(([label, n]) => {
    normalized[label] = Number((n * scale).toFixed(1));
  });
  return normalized;
}

function xrayRiskLevel(abnormalProbability) {
  const n = Number(abnormalProbability);
  if (!Number.isFinite(n)) return "Moderate";
  if (n < 25) return "Low";
  if (n < 50) return "Moderate";
  if (n < 75) return "High";
  return "Very High";
}

function renderGroupedLists(prefix, groups) {
  const setList = (idSuffix, items, fallback) => {
    const el = document.getElementById(`${prefix}${idSuffix}`);
    if (!el) return;
    el.innerHTML = "";
    const payload = items && items.length ? items : [fallback];
    payload.forEach((line) => {
      const li = document.createElement("li");
      li.textContent = line;
      el.appendChild(li);
    });
  };
  setList("NextSteps", groups.nextSteps, "Clinical follow-up planning should be based on physician review.");
  setList("RiskFactors", groups.riskFactors, "Risk factors were limited in the submitted inputs.");
  setList("PreventiveAdvice", groups.preventive, "Maintain regular screening and routine respiratory follow-up.");
  setList("WhyResult", groups.whyResult, "Result reflects model-estimated patterns from available data.");
}

function inferEhrRiskFactors(payload) {
  if (!payload) return [];
  const factors = [];
  const packYears = Number(payload.pack_years || 0);
  if (packYears >= 20) factors.push(`Smoking history (${packYears} pack-years)`);
  else if (packYears > 0) factors.push(`Smoking exposure (${packYears} pack-years)`);
  if (payload.copd_diagnosis === "Yes") factors.push("COPD diagnosis");
  if (payload.radon_exposure && payload.radon_exposure !== "Low") factors.push(`Radon exposure (${payload.radon_exposure})`);
  if (payload.asbestos_exposure === "Yes") factors.push("Asbestos exposure history");
  if (payload.secondhand_smoke_exposure === "Yes") factors.push("Secondhand smoke exposure");
  if (payload.family_history === "Yes") factors.push("Family history of lung cancer");
  return factors;
}

function getReportPatientMeta() {
  const nameEl = document.getElementById("reportPatientName");
  const ageEl = document.getElementById("reportPatientAge");
  const name = (nameEl && nameEl.value.trim()) || "—";
  const age = (ageEl && ageEl.value.trim()) || "—";
  return { name, age };
}

function formatReportDateTime() {
  return new Date().toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function showReportHead(el) {
  if (!el) return;
  el.classList.remove("hidden");
  el.setAttribute("aria-hidden", "false");
}

function hideReportHead(el) {
  if (!el) return;
  el.classList.add("hidden");
  el.setAttribute("aria-hidden", "true");
  el.innerHTML = "";
}

function fillClinicalReportHead(headEl, options) {
  if (!headEl) return;
  const {
    documentType,
    severityClass,
    severityNarrative,
    impression,
    detailParagraph,
    ancillaryLines = [],
    metricBar = null,
  } = options;
  const { name, age } = getReportPatientMeta();
  const when = formatReportDateTime();
  const allowed = ["low", "moderate", "high", "critical", "elevated"];
  const safeSeverity = allowed.includes(severityClass) ? severityClass : "moderate";

  const ancillaryHtml =
    ancillaryLines.length > 0
      ? `<ul class="clinical-report-head__ancillary">${ancillaryLines
          .map((line) => `<li>${escapeHtml(line)}</li>`)
          .join("")}</ul>`
      : "";

  const pct =
    metricBar && typeof metricBar.percent === "number"
      ? Math.min(100, Math.max(0, metricBar.percent))
      : null;
  const metricHtml =
    pct != null
      ? `<div class="clinical-report-metric">
          <div class="clinical-report-metric__hdr">
            <span class="clinical-report-metric__label">${escapeHtml(metricBar.label)}</span>
            <span class="clinical-report-metric__value">${escapeHtml(String(Math.round(pct)))}%</span>
          </div>
          <div class="clinical-report-metric__track">
            <div class="clinical-report-metric__fill" style="width:${pct}%"></div>
          </div>
        </div>`
      : "";

  headEl.innerHTML = `
    <header class="clinical-report-head__masthead">
      <div class="clinical-report-head__institution">Respiratory Health AI</div>
      <div class="clinical-report-head__doc-subtitle">Automated screening summary — research / educational prototype only</div>
    </header>
    <hr class="clinical-report-head__rule" />
    <table class="clinical-report-head__id-table">
      <tbody>
        <tr>
          <th scope="row">Patient name</th>
          <td>${escapeHtml(name)}</td>
          <th scope="row">Age (report)</th>
          <td>${escapeHtml(age)}</td>
        </tr>
        <tr>
          <th scope="row">Report date</th>
          <td colspan="3">${escapeHtml(when)}</td>
        </tr>
      </tbody>
    </table>
    <div class="clinical-report-head__doc-type">${escapeHtml(documentType)}</div>
    <div class="clinical-report-severity clinical-report-severity--${safeSeverity}">
      <div class="clinical-report-severity__cap">Automated stratification (software output)</div>
      <div class="clinical-report-severity__text">${escapeHtml(severityNarrative)}</div>
    </div>
    ${metricHtml}
    <section class="clinical-report-head__block">
      <h4 class="clinical-report-head__block-label">Impression</h4>
      <p class="clinical-report-head__block-text">${escapeHtml(impression)}</p>
    </section>
    <section class="clinical-report-head__block">
      <h4 class="clinical-report-head__block-label">Details</h4>
      <p class="clinical-report-head__block-text">${escapeHtml(detailParagraph)}</p>
    </section>
    ${ancillaryHtml}
    <p class="clinical-report-head__disclaimer">This output is produced by non-validated research software. It is not a medical device and does not constitute medical advice, diagnosis, or treatment. Qualified clinical review and standard-of-care workup remain required.</p>
  `;
  showReportHead(headEl);
}

function printClinicalReport(sectionEl) {
  if (!sectionEl || sectionEl.classList.contains("hidden")) return;
  sectionEl.classList.add("print-focus");
  document.body.classList.add("printing-report");
  let cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    sectionEl.classList.remove("print-focus");
    document.body.classList.remove("printing-report");
  };
  window.addEventListener("afterprint", cleanup, { once: true });
  window.print();
  window.setTimeout(cleanup, 1500);
}

/** Sample EHR rows for quick testing (values must match model metadata). */
const EHR_DUMMY_PRESETS = [
  { age: 42, pack_years: 8, gender: "Male", radon_exposure: "Low", asbestos_exposure: "No", secondhand_smoke_exposure: "No", copd_diagnosis: "No", alcohol_consumption: "Moderate", family_history: "No" },
  { age: 58, pack_years: 32, gender: "Female", radon_exposure: "Medium", asbestos_exposure: "No", secondhand_smoke_exposure: "Yes", copd_diagnosis: "No", alcohol_consumption: "None", family_history: "Yes" },
  { age: 67, pack_years: 45, gender: "Male", radon_exposure: "High", asbestos_exposure: "Yes", secondhand_smoke_exposure: "No", copd_diagnosis: "Yes", alcohol_consumption: "Heavy", family_history: "No" },
  { age: 51, pack_years: 18, gender: "Female", radon_exposure: "Low", asbestos_exposure: "No", secondhand_smoke_exposure: "No", copd_diagnosis: "No", alcohol_consumption: "Moderate", family_history: "No" },
  { age: 73, pack_years: 52, gender: "Male", radon_exposure: "Medium", asbestos_exposure: "Yes", secondhand_smoke_exposure: "Yes", copd_diagnosis: "Yes", alcohol_consumption: "Moderate", family_history: "Yes" },
  { age: 39, pack_years: 0, gender: "Female", radon_exposure: "Low", asbestos_exposure: "No", secondhand_smoke_exposure: "No", copd_diagnosis: "No", alcohol_consumption: "None", family_history: "No" },
  { age: 62, pack_years: 28, gender: "Male", radon_exposure: "High", asbestos_exposure: "No", secondhand_smoke_exposure: "No", copd_diagnosis: "No", alcohol_consumption: "Heavy", family_history: "Yes" },
  { age: 55, pack_years: 22, gender: "Female", radon_exposure: "Medium", asbestos_exposure: "No", secondhand_smoke_exposure: "Yes", copd_diagnosis: "No", alcohol_consumption: "Moderate", family_history: "No" },
  { age: 48, pack_years: 12, gender: "Male", radon_exposure: "Low", asbestos_exposure: "Yes", secondhand_smoke_exposure: "No", copd_diagnosis: "No", alcohol_consumption: "Moderate", family_history: "No" },
  { age: 69, pack_years: 38, gender: "Female", radon_exposure: "High", asbestos_exposure: "No", secondhand_smoke_exposure: "No", copd_diagnosis: "Yes", alcohol_consumption: "None", family_history: "Yes" },
  { age: 44, pack_years: 6, gender: "Male", radon_exposure: "Medium", asbestos_exposure: "No", secondhand_smoke_exposure: "Yes", copd_diagnosis: "No", alcohol_consumption: "Heavy", family_history: "No" },
  { age: 76, pack_years: 60, gender: "Male", radon_exposure: "High", asbestos_exposure: "Yes", secondhand_smoke_exposure: "Yes", copd_diagnosis: "Yes", alcohol_consumption: "Moderate", family_history: "Yes" },
  { age: 33, pack_years: 3, gender: "Female", radon_exposure: "Low", asbestos_exposure: "No", secondhand_smoke_exposure: "No", copd_diagnosis: "No", alcohol_consumption: "Moderate", family_history: "No" },
  { age: 59, pack_years: 25, gender: "Female", radon_exposure: "Medium", asbestos_exposure: "Yes", secondhand_smoke_exposure: "No", copd_diagnosis: "No", alcohol_consumption: "None", family_history: "No" },
  { age: 64, pack_years: 40, gender: "Male", radon_exposure: "Low", asbestos_exposure: "No", secondhand_smoke_exposure: "Yes", copd_diagnosis: "Yes", alcohol_consumption: "Heavy", family_history: "No" },
];

const EHR_FORM_FIELD_NAMES = [
  "age",
  "pack_years",
  "gender",
  "radon_exposure",
  "asbestos_exposure",
  "secondhand_smoke_exposure",
  "copd_diagnosis",
  "alcohol_consumption",
  "family_history",
];

function clearEhrInputsInForm(form) {
  if (!form) return;
  EHR_FORM_FIELD_NAMES.forEach((name) => {
    const el = form.elements[name];
    if (!el) return;
    if (el.tagName === "INPUT") el.value = "";
    if (el.tagName === "SELECT") el.value = "";
  });
}

function applyEhrPresetToForm(form, preset) {
  if (!form || !preset) return;
  EHR_FORM_FIELD_NAMES.forEach((name) => {
    const el = form.elements[name];
    if (!el || preset[name] === undefined) return;
    el.value = String(preset[name]);
  });
}

function wireEhrFillButtons() {
  const buttons = document.querySelectorAll(".fill-dummy-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const form = btn.closest("form");
      if (!form) return;
      const pick =
        EHR_DUMMY_PRESETS[Math.floor(Math.random() * EHR_DUMMY_PRESETS.length)];
      applyEhrPresetToForm(form, pick);
      const msgEl = form.querySelector(".message");
      if (msgEl) setMessage(msgEl, "Sample EHR profile filled. You can edit any field.");
    });
  });
}

function renderProbGrid(container, probabilities, predicted) {
  container.innerHTML = "";
  const view = normalizedProbabilityView(probabilities);
  const entries = Object.entries(view).sort((a, b) => b[1] - a[1]);
  entries.forEach(([name, prob]) => {
    const meta = diseaseMeta(name);
    const item = document.createElement("div");
    item.className = "prob-item";
    if (name === predicted) item.classList.add("prob-top");
    item.innerHTML = `
      <div class="prob-row">
        <span><i class="fa-solid ${meta.icon}" style="color:${meta.color}"></i> ${name.replace("_", " ")}</span>
        <strong>${formatPercent(prob)}</strong>
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
const xrayReportHead = document.getElementById("xrayReportHead");
const resultTitle = document.getElementById("resultTitle");
const resultSubtitle = document.getElementById("resultSubtitle");
const resultIcon = document.getElementById("resultIcon");
const probGrid = document.getElementById("probGrid");
const xrayRecs = document.getElementById("xrayRecs");
const analysisHeatmap = document.getElementById("analysisHeatmap");
const analysisSteps = document.getElementById("analysisSteps");
const confidenceFill = document.getElementById("confidenceFill");
const xrayPrimaryFinding = document.getElementById("xrayPrimaryFinding");
const xrayRiskLevelEl = document.getElementById("xrayRiskLevel");
const xrayOverallConfidence = document.getElementById("xrayOverallConfidence");
const xrayAnalysisNarrative = document.getElementById("xrayAnalysisNarrative");
const xrayAbnormalityNarrative = document.getElementById("xrayAbnormalityNarrative");

function resetXrayResult() {
  resultPanel.classList.add("hidden");
  resultPanel.classList.remove("result-normal", "result-abnormal");
  resultIcon.className = "fa-solid fa-circle-info";
  hideReportHead(xrayReportHead);
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
        `Overall confidence ${formatPercent(data.confidence)}`;
      if (confidenceFill) confidenceFill.style.width = `${Math.min(100, data.confidence)}%`;
      if (xrayPrimaryFinding) xrayPrimaryFinding.textContent = data.prediction.replace("_", " ");
      if (xrayRiskLevelEl) xrayRiskLevelEl.textContent = xrayRiskLevel(data.abnormal_probability);
      if (xrayOverallConfidence) xrayOverallConfidence.textContent = formatPercent(data.confidence);
      if (xrayAnalysisNarrative) {
        xrayAnalysisNarrative.textContent = `Findings suggest a high likelihood of ${data.prediction.replace("_", " ")} (${formatPercent(data.confidence)} confidence).`;
      }
      if (xrayAbnormalityNarrative) {
        xrayAbnormalityNarrative.textContent = `Abnormality detected: ${xrayRiskLevel(data.abnormal_probability)} (${formatPercent(data.abnormal_probability)} composite abnormal signal).`;
      }

      const predLabel = data.prediction.replace("_", " ");
      fillClinicalReportHead(xrayReportHead, {
        documentType: "CHEST RADIOGRAPH — AUTOMATED CLASSIFICATION SUMMARY",
        severityClass: data.prediction === "Normal" ? "low" : "elevated",
        severityNarrative:
          data.prediction === "Normal"
            ? "Software emphasis: pattern weighted toward non-acute categories for this exposure. Clinical correlation still required."
            : "Software emphasis: pattern warrants correlation with history, examination, and standard imaging interpretation. Not diagnostic.",
        impression: `Primary predicted category: ${predLabel}.`,
        detailParagraph: `Classifier confidence ${formatPercent(data.confidence)}. Aggregate probability assigned to non-Normal categories (abnormal signal): ${formatPercent(data.abnormal_probability)}.`,
        ancillaryLines: [],
        metricBar: {
          label: "Stated classifier confidence",
          percent: clampPercent(data.confidence),
        },
      });

      if (probGrid) renderProbGrid(probGrid, data.probabilities, data.prediction);

      renderGroupedLists("xray", {
        nextSteps: (data.recommendations || []).slice(0, 3),
        riskFactors: [
          `Primary radiographic pattern: ${data.prediction.replace("_", " ")}`,
          `Model-derived abnormality burden: ${formatPercent(data.abnormal_probability)}`,
        ],
        preventive: [
          "Repeat imaging or laboratory confirmation should follow clinician judgment.",
          "Monitor respiratory symptoms and seek urgent care if worsening occurs.",
        ],
        whyResult: [
          `Key visual pattern aligned most strongly with ${data.prediction.replace("_", " ")}.`,
          `The model assigned ${formatPercent(data.confidence)} confidence to the top class relative to alternatives.`,
        ],
      });

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

document.getElementById("xrayPrintBtn")?.addEventListener("click", () => {
  printClinicalReport(resultPanel);
});

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
const ehrReportHead = document.getElementById("ehrReportHead");
const ehrPrimaryFinding = document.getElementById("ehrPrimaryFinding");
const ehrRiskLevelEl = document.getElementById("ehrRiskLevel");
const ehrOverallConfidence = document.getElementById("ehrOverallConfidence");
const ehrAnalysisNarrative = document.getElementById("ehrAnalysisNarrative");
const ehrContributingFactors = document.getElementById("ehrContributingFactors");
let lastEhrPayload = null;

function applyRiskBand(panel, tier) {
  panel.classList.remove("risk-low", "risk-moderate", "risk-high", "risk-critical");
  panel.classList.add(`risk-${tier}`);
}

function renderRiskResult(data) {
  ehrResult.classList.remove("hidden");
  applyRiskBand(ehrResult, data.risk_band.tier);
  fillClinicalReportHead(ehrReportHead, {
    documentType: "STRUCTURED CLINICAL DATA — TABULAR RISK ESTIMATE (LUNG CANCER MODEL)",
    severityClass: data.risk_band.tier,
    severityNarrative: `${data.risk_band.label}. Model-estimated probability ${formatPercent(data.probability)} (reference ${data.risk_band.range}).`,
    impression:
      "Estimated probability from a fixed-feature tabular model; intended for workflow demonstration only.",
    detailParagraph: `Risk band per internal thresholds: ${data.risk_band.label}. Output probability: ${formatPercent(data.probability)}.`,
    ancillaryLines: [`Software tier label: ${data.risk_band.tier}`],
    metricBar: {
      label: "Model-estimated probability",
      percent: clampPercent(data.probability),
    },
  });
  ehrResultTitle.textContent = `${data.risk_band.label} · ${formatPercent(data.probability)}`;
  ehrResultSubtitle.textContent = `Estimated lung cancer probability (range ${data.risk_band.range}).`;
  ehrRiskFill.style.width = `${Math.min(100, Math.max(3, data.probability))}%`;
  const inferredFactors = inferEhrRiskFactors(lastEhrPayload);
  if (ehrPrimaryFinding) ehrPrimaryFinding.textContent = "Lung cancer risk estimation";
  if (ehrRiskLevelEl) ehrRiskLevelEl.textContent = data.risk_band.label;
  if (ehrOverallConfidence) ehrOverallConfidence.textContent = formatPercent(data.probability);
  if (ehrAnalysisNarrative) ehrAnalysisNarrative.textContent = `Findings suggest ${data.risk_band.label.toLowerCase()} based on structured clinical inputs.`;
  if (ehrContributingFactors) {
    ehrContributingFactors.textContent = inferredFactors.length
      ? `Contributing factors: ${inferredFactors.join(", ")}.`
      : "Contributing factors: limited high-risk structured factors were reported.";
  }
  renderGroupedLists("ehr", {
    nextSteps: (data.recommendations || []).slice(0, 3),
    riskFactors: inferredFactors,
    preventive: [
      "Review smoking cessation and environmental risk mitigation strategies.",
      "Plan interval follow-up based on clinician assessment and local protocols.",
    ],
    whyResult: [
      `The tabular model integrated demographic and exposure variables to estimate ${formatPercent(data.probability)} risk.`,
      "Risk band assignment follows predefined internal thresholds.",
    ],
  });
}

if (ehrForm) {
  ehrForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(ehrForm).entries());
    lastEhrPayload = payload;
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
      hideReportHead(ehrReportHead);
    }
  });
}

document.getElementById("ehrPrintBtn")?.addEventListener("click", () => {
  printClinicalReport(ehrResult);
});

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
const combinedReportHead = document.getElementById("combinedReportHead");
const combinedPrimaryFinding = document.getElementById("combinedPrimaryFinding");
const combinedRiskLevelEl = document.getElementById("combinedRiskLevel");
const combinedOverallConfidence = document.getElementById("combinedOverallConfidence");
const combinedReasoningText = document.getElementById("combinedReasoningText");
let lastCombinedPayload = null;

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
    setMessage(
      combinedMessage,
      "Image ready. Add EHR fields for multimodal fusion, or submit with X-ray only."
    );
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
    lastCombinedPayload = Object.fromEntries(formData.entries());
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

      combinedTitle.textContent = `${data.combined.risk_band.label} · Score ${formatPercent(data.combined.score)}`;
      const ehrUsed = data.combined.ehr_used !== false && data.lung_cancer;
      combinedSubtitle.textContent = ehrUsed
        ? "Fused multi-disease X-ray finding + EHR-derived lung cancer risk."
        : "X-ray abnormality score (EHR optional — not applied for this run).";

      fillClinicalReportHead(combinedReportHead, {
        documentType: "MULTIMODAL SUMMARY — RADIOGRAPH + STRUCTURED FACTORS (FUSED SCORE)",
        severityClass: data.combined.risk_band.tier,
        severityNarrative: `Composite score ${formatPercent(data.combined.score)}. ${data.combined.risk_band.label}. Fusion method: ${data.combined.method}.`,
        impression: ehrUsed
          ? "Combined assessment incorporates radiograph-derived signal and structured clinical inputs submitted in this form."
          : "Combined assessment uses radiograph-derived signal only; structured clinical inputs were not applied to the fusion.",
        detailParagraph: `Radiograph predicted category: ${data.xray.prediction.replace("_", " ")} (confidence ${formatPercent(data.xray.confidence)}). Abnormal signal: ${formatPercent(data.xray.abnormal_probability)}. ${
          ehrUsed
            ? `Structured-model probability included: ${formatPercent(data.lung_cancer.probability)}.`
            : "Structured-model probability not included in fusion for this run."
        }`,
        ancillaryLines: [],
        metricBar: {
          label: "Fused composite score (display scale 0–100)",
          percent: clampPercent(data.combined.score),
        },
      });
      if (combinedPrimaryFinding) combinedPrimaryFinding.textContent = data.xray.prediction.replace("_", " ");
      if (combinedRiskLevelEl) combinedRiskLevelEl.textContent = data.combined.risk_band.label;
      if (combinedOverallConfidence) combinedOverallConfidence.textContent = formatPercent(data.combined.score);
      if (combinedReasoningText) {
        combinedReasoningText.textContent = ehrUsed
          ? `Combined score ${formatPercent(data.combined.score)} reflects radiographic abnormality and submitted clinical risk factors.`
          : `Combined score ${formatPercent(data.combined.score)} reflects radiographic findings only because EHR factors were not complete.`;
      }

      combinedXrayEl.textContent =
        `${data.xray.prediction.replace("_", " ")} (${formatPercent(data.xray.confidence)})`;
      if (combinedXrayDetail)
        combinedXrayDetail.textContent =
          `Abnormal signal: ${formatPercent(data.xray.abnormal_probability)}`;
      combinedEhrEl.textContent = ehrUsed
        ? `${formatPercent(data.lung_cancer.probability)} · ${data.lung_cancer.risk_band.label}`
        : "— (not used)";
      combinedOverallEl.textContent =
        `${formatPercent(data.combined.score)} · ${data.combined.risk_band.label}`;

      renderProbGrid(combinedProbGrid, data.xray.probabilities, data.xray.prediction);

      const combinedFactors = [
        `X-ray finding weighted highest: ${data.xray.prediction.replace("_", " ")}`,
        ...inferEhrRiskFactors(lastCombinedPayload),
      ];
      renderGroupedLists("combined", {
        nextSteps: [
          ...(data.combined.summary || []).slice(0, 2),
          ...((data.xray.recommendations || []).slice(0, 1)),
          ...(ehrUsed && data.lung_cancer.recommendations ? data.lung_cancer.recommendations.slice(0, 1) : []),
        ],
        riskFactors: combinedFactors,
        preventive: [
          "Schedule longitudinal follow-up to monitor radiographic and risk-factor progression.",
          "Use clinician-led counseling for modifiable exposures and smoking risk.",
        ],
        whyResult: [
          `X-ray pattern indicates ${data.xray.prediction.replace("_", " ")} with ${formatPercent(data.xray.confidence)} confidence.`,
          ehrUsed
            ? `EHR factors shifted combined risk estimate to ${formatPercent(data.combined.score)}.`
            : "EHR factors were incomplete, so combined score tracks X-ray signal only.",
        ],
      });

      setMessage(combinedMessage, "Combined assessment complete.");
    } catch (err) {
      setMessage(combinedMessage, err.message || "Something went wrong.", true);
      combinedResult.classList.add("hidden");
      hideReportHead(combinedReportHead);
    } finally {
      combinedBtn.disabled = false;
      combinedBtn.innerHTML =
        '<i class="fa-solid fa-layer-group"></i> Run Combined Assessment';
    }
  });
}

document.getElementById("combinedPrintBtn")?.addEventListener("click", () => {
  printClinicalReport(combinedResult);
});

wireEhrFillButtons();
