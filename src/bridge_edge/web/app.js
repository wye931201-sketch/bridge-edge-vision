"use strict";
const $ = (id) => document.getElementById(id);
let selectedFile = null, previewUrl = null, busy = false, modelReady = false, classes = [];
function error(message) { $("error").textContent = message; $("error").classList.toggle("hidden", !message); }
function syncButton() { $("detect").disabled = busy || !selectedFile || !modelReady; }
function chooseFile(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) { error("请选择 JPG、PNG 或 WEBP 图片。"); return; }
  if (file.size > 25 * 1024 * 1024) { error("图片超过 25 MiB，请先缩小图片。"); return; }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file); $("preview").src = previewUrl;
  $("filename").textContent = file.name; $("image-label").textContent = "原始图片";
  $("dropzone").classList.add("hidden"); $("image-wrap").classList.remove("hidden"); $("replace").classList.remove("hidden");
  $("result-body").classList.add("hidden"); $("empty-result").classList.remove("hidden"); $("result-status").textContent = "待运行";
  error(""); syncButton();
}
$("file").addEventListener("change", (event) => chooseFile(event.target.files[0]));
$("replace").addEventListener("click", () => $("file").click());
$("preview").addEventListener("load", () => { $("dimensions").textContent = `${$("preview").naturalWidth} × ${$("preview").naturalHeight} px · 本机处理`; });
for (const eventName of ["dragenter", "dragover"]) $("dropzone").addEventListener(eventName, (event) => { event.preventDefault(); $("dropzone").classList.add("dragover"); });
for (const eventName of ["dragleave", "drop"]) $("dropzone").addEventListener(eventName, (event) => { event.preventDefault(); $("dropzone").classList.remove("dragover"); });
$("dropzone").addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));
$("confidence").addEventListener("input", () => { $("confidence-value").textContent = Number($("confidence").value).toFixed(2); });

async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "请求未完成，请查看本地服务日志。");
  return value;
}
function showResult(result) {
  $("preview").src = result.annotated_url;
  $("image-wrap").classList.remove("hidden"); $("dropzone").classList.add("hidden");
  $("image-label").textContent = "模型检测结果"; $("filename").textContent = result.source_filename;
  $("result-body").classList.remove("hidden"); $("empty-result").classList.add("hidden");
  $("result-status").textContent = result.backend === "onnx" ? "ONNX · CPU" : "PyTorch · 本地";
  $("finding-count").textContent = result.finding_count; $("latency").textContent = Math.round(result.total_latency_ms);
  $("findings").replaceChildren();
  for (const finding of result.detections) {
    const row = document.createElement("div"); row.className = "finding";
    const dot = document.createElement("i"), name = document.createElement("span"), score = document.createElement("span");
    const color = classes.find((item) => item.id === finding.class_id)?.color || [8, 127, 131];
    dot.style.background = `rgb(${color.join(",")})`; name.textContent = finding.chinese; score.textContent = finding.confidence.toFixed(2); score.className = "score";
    row.append(dot, name, score); row.title = `检测框：${finding.xyxy.map(Math.round).join(", ")} px`; $("findings").append(row);
  }
  if (!result.finding_count) { const note = document.createElement("p"); note.className = "muted"; note.textContent = "当前阈值下未检出候选。可调整阈值或复核原图。"; $("findings").append(note); }
  $("download-image").href = result.annotated_url; $("download-json").href = result.json_url;
}
$("detect").addEventListener("click", async () => {
  if (!selectedFile || busy) return;
  busy = true; syncButton(); error(""); $("detect").firstElementChild.textContent = "正在本机处理…"; $("detect").classList.add("busy");
  try {
    const query = new URLSearchParams({filename: selectedFile.name, model: $("model").value, confidence: $("confidence").value, tile: $("tile").checked});
    const result = await jsonRequest(`/api/predict?${query}`, {method: "POST", headers: {"Content-Type": selectedFile.type}, body: selectedFile});
    showResult(result); await loadHistory();
  } catch (failure) { error(failure.message); }
  finally { busy = false; syncButton(); $("detect").firstElementChild.textContent = "开始检测"; $("detect").classList.remove("busy"); }
});

async function loadHistory() {
  const data = await jsonRequest("/api/history"); $("history").replaceChildren();
  if (!data.items.length) { const note = document.createElement("p"); note.className = "muted"; note.textContent = "尚无检测记录"; $("history").append(note); }
  for (const item of data.items.slice(0, 8)) {
    const button = document.createElement("button"); button.className = "history-item";
    const title = document.createElement("strong"), meta = document.createElement("small"); title.textContent = item.source_filename;
    meta.textContent = `${item.finding_count} 个候选 · ${Math.round(item.latency_ms)} ms · ${new Date(item.created_utc).toLocaleTimeString("zh-CN", {hour: "2-digit", minute: "2-digit"})}`;
    button.append(title, meta); button.addEventListener("click", async () => {
      if (busy) return;
      try {
        showResult(await jsonRequest(`/api/history/${item.id}`));
        selectedFile = null; $("replace").classList.remove("hidden"); syncButton();
      } catch (failure) { error(failure.message); }
    }); $("history").append(button);
  }
}
$("refresh-history").addEventListener("click", () => loadHistory().catch((failure) => error(failure.message)));
async function start() {
  const data = await jsonRequest("/api/health"); classes = data.classes;
  $("connection").textContent = data.hardware.cuda_available ? `${data.hardware.device} · 本地就绪` : "CPU · 本地引擎";
  $("model").replaceChildren();
  for (const entry of data.models) { const option = document.createElement("option"); option.value = entry.id; option.textContent = `${entry.variant || "detector"} · ${entry.scope === "pilot" ? "预实验" : entry.scope || "本地模型"}`; $("model").append(option); }
  if (data.default_model) $("model").value = data.default_model;
  modelReady = data.models.length > 0;
  if (!modelReady) { const option = document.createElement("option"); option.textContent = "尚无训练权重"; $("model").append(option); $("model-note").textContent = "请先完成数据准备和一次本地训练。"; }
  syncButton();
  const sampleData = await jsonRequest("/api/samples");
  for (const sample of sampleData.samples) {
    const button = document.createElement("button"), image = document.createElement("img"); button.className = "sample"; button.title = sample.id; image.alt = "验证集原始样例"; image.src = sample.url; image.loading = "lazy"; button.append(image);
    button.addEventListener("click", async () => { try { const response = await fetch(sample.url); chooseFile(new File([await response.blob()], sample.id, {type: "image/jpeg"})); } catch (failure) { error(failure.message); } }); $("samples").append(button);
  }
  await loadHistory();
}
start().catch((failure) => { $("connection").textContent = "本地服务未就绪"; error(failure.message); });
