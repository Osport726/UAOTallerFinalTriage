/**
 * triage.js — Lógica de interacción para la UI de Triage de Soporte TI.
 *
 * Responsabilidad única: gestionar la selección de imagen, enviar la petición
 * al endpoint /api/predict, actualizar la UI con el resultado y permitir
 * la descarga del reporte PDF via /api/report.
 *
 * Flujo principal:
 *   1. Usuario selecciona/arrastra una imagen → handleFile()
 *   2. Clic en "Analizar" → fetch POST /api/predict → showResult()
 *   3. Clic en "Descargar PDF" → fetch POST /api/report → descarga automática
 */

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewArea = document.getElementById("preview-area");
const previewImg = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const analyzeBtn = document.getElementById("analyze-btn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error-box");
const resultCard = document.getElementById("result");
const resetBtn = document.getElementById("reset-btn");
const pdfBtn = document.getElementById("pdf-btn");

/** @type {File|null} Archivo de imagen actualmente seleccionado. */
let selectedFile = null;

// Selección de archivo

/**
 * Actualiza la vista previa y habilita el botón de análisis.
 * @param {File} file - Archivo de imagen seleccionado por el usuario.
 */
function handleFile(file) {
  if (!file) return;
  selectedFile = file;

  previewImg.src = URL.createObjectURL(file);
  previewName.textContent =
    file.name + " (" + (file.size / 1024).toFixed(1) + " KB)";
  previewArea.style.display = "block";
  analyzeBtn.disabled = false;
  pdfBtn.disabled = true;
  hideResult();
  hideError();
}

fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("dragover")
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

// Análisis

/**
 * Envía la imagen a /api/predict y renderiza el resultado.
 * Habilita el botón PDF una vez que el análisis es exitoso.
 */
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  pdfBtn.disabled = true;
  loading.style.display = "block";
  hideResult();
  hideError();

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.detail || "Error al procesar la imagen.");
      return;
    }

    showResult(data);
    pdfBtn.disabled = false;  // habilitar PDF solo si el análisis fue exitoso
  } catch {
    showError("No se pudo conectar con el servidor. Verifique que esté corriendo.");
  } finally {
    loading.style.display = "none";
    analyzeBtn.disabled = false;
  }
});

// Descarga PDF

/**
 * Envía la imagen a /api/report y descarga automáticamente el PDF resultante.
 * Usa fetch + Blob para no redirigir la página.
 */
pdfBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  pdfBtn.disabled = true;
  pdfBtn.textContent = "Generando PDF...";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/api/report", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      showError(err.detail || "No se pudo generar el reporte PDF.");
      return;
    }

    // Obtener nombre del archivo del header Content-Disposition
    const disposition = response.headers.get("content-disposition") || "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : "reporte_triage.pdf";

    // Descargar como Blob sin recargar la página
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    showError("No se pudo descargar el reporte. Verifique la conexión.");
  } finally {
    pdfBtn.disabled = false;
    pdfBtn.textContent = "⬇ Descargar reporte PDF";
  }
});

// Reset

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewArea.style.display = "none";
  previewImg.src = "";
  analyzeBtn.disabled = true;
  pdfBtn.disabled = true;
  hideResult();
  hideError();
});

// Helpers

/**
 * Renderiza el resultado del triage en la tarjeta de resultado.
 * @param {Object} data - Respuesta JSON de /api/predict.
 * @param {Array}  data.predictions - Lista de predicciones top-k.
 * @param {Object} data.model - Información del modelo (min_confidence, etc.).
 * @param {string} data.timestamp - ISO timestamp de la clasificación.
 */
function showResult(data) {
  const top = data.predictions[0];
  const minConfidence = data.model?.min_confidence ?? 0.40;

  document.getElementById("res-category").textContent = top.category;
  document.getElementById("res-team").textContent = top.team;

  const pct = Math.round(top.score * 100);
  document.getElementById("res-score").textContent = pct + "%";

  const fill = document.getElementById("score-bar-fill");
  fill.style.width = pct + "%";
  fill.className =
    "bar-fill" + (pct < 40 ? " very-low" : pct < 65 ? " low" : "");

  const ts = new Date(data.timestamp);
  document.getElementById("res-timestamp").textContent =
    ts.toLocaleDateString("es-CO") + " " + ts.toLocaleTimeString("es-CO");

  const alertHuman = document.getElementById("alert-human");
  alertHuman.style.display = top.score < minConfidence ? "block" : "none";

  resultCard.style.display = "block";
}

function hideResult() {
  resultCard.style.display = "none";
}

/**
 * Muestra un mensaje de error en el banner de error.
 * @param {string} msg - Texto del error a mostrar.
 */
function showError(msg) {
  errorBox.textContent = "Error: " + msg;
  errorBox.style.display = "block";
}

function hideError() {
  errorBox.style.display = "none";
}
