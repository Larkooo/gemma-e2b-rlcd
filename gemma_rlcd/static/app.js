"use strict";

const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const typeNames = {choice: "Choice", score: "Grade", independent: "Independent labels", noul: "Yes / no"};
const typeNotes = {
  choice: "One winner. Candidate probabilities sum to 100%.",
  score: "Levels run from 0 upward. The result is the probability-weighted grade.",
  independent: "Each label gets its own yes probability; values do not need to sum to 100%.",
  noul: "Returns the probability that the proposition is true.",
};
const presets = {
  animals: {text: "A cat sleeps on the sofa. No dogs are present.", instructions: "", questions: {
    animal: {type: "choice", instructions: "Which animal is present?", criteria: {cat: "A cat", dog: "A dog", other: "Neither a cat nor a dog"}},
    presence: {type: "independent", instructions: "Which animals are present in the evidence?", criteria: {cat: "A cat is present", dog: "A dog is present"}},
  }},
  video: {text: "", instructions: "Judge visual evidence in the sampled frames. A spoken mention does not count as a visible animal.", questions: {
    dog_concentration: {type: "score", instructions: "How often are dogs visible in the sampled video frames?", criteria: ["No sampled frames contain a visible dog", "Visible in at most one quarter of sampled frames", "Visible in about half of sampled frames", "Visible in most sampled frames", "Visible in every sampled frame"]},
    cat_concentration: {type: "score", instructions: "How often are cats visible in the sampled video frames?", criteria: ["No sampled frames contain a visible cat", "Visible in at most one quarter of sampled frames", "Visible in about half of sampled frames", "Visible in most sampled frames", "Visible in every sampled frame"]},
  }},
  sentiment: {text: "The delivery was a day late, but the headphones sound wonderful and the support team fixed the issue quickly.", instructions: "", questions: {
    sentiment: {type: "choice", instructions: "What is the overall sentiment of this review?", criteria: {positive: "Mostly positive", mixed: "Mixed positive and negative", negative: "Mostly negative"}},
    satisfaction: {type: "score", instructions: "How satisfied is the customer?", criteria: ["Very dissatisfied", "Somewhat dissatisfied", "Neutral or mixed", "Mostly satisfied", "Very satisfied"]},
  }},
  visual: {text: "", instructions: "Base your answers only on the supplied image.", questions: {
    dominant_color: {type: "choice", instructions: "Which color is most prominent?", criteria: {red: "Red", blue: "Blue", green: "Green", other: "Another color"}},
    red_coverage: {type: "score", instructions: "How much of the image is red?", criteria: ["No red is visible", "Red covers a small part", "Red covers about half", "Red covers most or all of the image"]},
  }},
  speech: {text: "", instructions: "Listen to the audio and classify what the speaker says.", questions: {
    intent: {type: "choice", instructions: "What is the speaker's main intent?", criteria: {question: "Asking a question", request: "Requesting an action", statement: "Making a statement", other: "Something else or no intelligible speech"}},
    dog_mentioned: {type: "noul", instructions: "Does the speaker mention a dog?", criteria: {true: "The speech mentions a dog", false: "The speech does not mention a dog"}},
  }},
  blank: {text: "", instructions: "", questions: {answer: {type: "choice", instructions: "", criteria: {option_a: "", option_b: ""}}}},
};

let fields = [];
let attachments = [];
let result = null;
let modelReady = false;
let busy = false;
let recorder = null;
let recordingStream = null;
let recordingTimer = null;
let draftTimer = null;
let activeView = "answers";

function showError(message) {
  $("#error").textContent = message;
  $("#error").hidden = !message;
}

function fieldsFromSchema(questions) {
  return Object.entries(questions).map(([name, question]) => ({
    id: crypto.randomUUID(), name, type: question.type, instructions: question.instructions,
    criteria: question.type === "score"
      ? question.criteria.map((description, index) => ({name: String(index), description}))
      : Object.entries(question.criteria || {true: "The proposition is true", false: "The proposition is false"}).map(([key, description]) => ({name: key, description})),
  }));
}

function buildSchema(validate = true) {
  const questions = Object.create(null);
  for (const field of fields) {
    const name = field.name.trim();
    if (validate && (!name || Object.hasOwn(questions, name))) throw new Error("Give every output field a unique, nonempty name.");
    if (validate && !field.instructions.trim()) throw new Error(`Add a question or instruction for “${name}”.`);
    let criteria;
    if (field.type === "score") {
      criteria = field.criteria.map((row, index) => row.description.trim() ? row.description : `Grade ${index} on a scale from 0 (lowest) to ${field.criteria.length - 1} (highest)`);
    } else {
      criteria = Object.create(null);
      for (const row of field.criteria) {
        const key = row.name.trim();
        if (validate && (!key || Object.hasOwn(criteria, key))) throw new Error(`Use unique, nonempty option names in “${name}”.`);
        criteria[key] = row.description.trim() ? row.description : key;
      }
    }
    questions[name] = {type: field.type, instructions: field.instructions, criteria};
  }
  return questions;
}

function markChanged() {
  if (result) {
    $("#result-badge").textContent = "Inputs changed";
    $("#result-badge").className = "result-badge stale";
  }
  updateCount();
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => {
    try { localStorage.setItem("decision-lab-draft-v1", JSON.stringify({text: $("#state-text").value, instructions: $("#instructions").value, fields})); }
    catch { /* The current workspace remains usable if browser storage is unavailable. */ }
  }, 250);
}

function updateCount() {
  const decisions = fields.reduce((count, field) => count + (field.type === "independent" ? field.criteria.length : 1), 0);
  $("#run-count").textContent = `${fields.length} output ${fields.length === 1 ? "field" : "fields"} · ${decisions} ${decisions === 1 ? "decision" : "decisions"}`;
}

function renderFields() {
  $("#fields").innerHTML = fields.map((field, fieldIndex) => {
    const isGrade = field.type === "score";
    const isBoolean = field.type === "noul";
    const minimum = field.type === "independent" ? 1 : 2;
    return `<article class="field" data-id="${field.id}">
      <div class="field-header"><input data-prop="name" value="${escapeHTML(field.name)}" aria-label="Field ${fieldIndex + 1} name" spellcheck="false"><select data-prop="type" aria-label="${escapeHTML(field.name || "Field")} output type">${Object.entries(typeNames).map(([type, label]) => `<option value="${type}" ${type === field.type ? "selected" : ""}>${label}</option>`).join("")}</select><button type="button" class="icon-button" data-action="remove-field" aria-label="Remove field ${escapeHTML(field.name)}" ${fields.length === 1 ? "disabled" : ""}>×</button></div>
      <div class="field-body"><label for="instruction-${field.id}">Question / instructions</label><textarea id="instruction-${field.id}" class="field-instruction" data-prop="instructions" rows="2" placeholder="What should this field evaluate?">${escapeHTML(field.instructions)}</textarea>
      <div class="criteria-label"><span>${isGrade ? "Grade scale" : isBoolean ? "Yes / no" : "Choices"}</span>${isGrade ? `<select data-prop="levelCount" aria-label="Number of grade levels for ${escapeHTML(field.name)}">${Array.from({length:9}, (_, i) => i + 2).map((count) => `<option value="${count}" ${count === field.criteria.length ? "selected" : ""}>0–${count - 1} (${count} levels)</option>`).join("")}</select>` : `<span>${field.criteria.length} options</span>`}</div>
      ${isGrade || isBoolean ? `<details class="optional-rubric"><summary>${isGrade ? "Customize level labels" : "Customize yes / no meanings"} <span class="subtle">optional</span></summary>` : ""}
      ${field.criteria.map((row, index) => `<div class="criterion ${isGrade ? "score-row" : isBoolean ? "boolean-row" : "choice-row"}" data-index="${index}">
        ${isGrade || isBoolean ? `<span class="criterion-label">${isGrade ? index : escapeHTML(row.name)}</span><textarea data-option="description" rows="1" aria-label="${isGrade ? "Grade level" : "Option"} ${isGrade ? index : index + 1} description for ${escapeHTML(field.name)}" placeholder="Optional label or description">${escapeHTML(row.description)}</textarea>` : `<div class="simple-choice"><input data-option="name" value="${escapeHTML(row.name)}" aria-label="Option ${index + 1} name for ${escapeHTML(field.name)}" placeholder="Choice, e.g. Porsche" spellcheck="false"><details><summary>${row.description.trim() && row.description !== row.name ? "Description added" : "Add description"} <span class="subtle">optional</span></summary><textarea data-option="description" rows="1" aria-label="Option ${index + 1} description for ${escapeHTML(field.name)}" placeholder="Only if this choice needs more context">${escapeHTML(row.description)}</textarea></details></div>`}
        ${isGrade ? `<button type="button" class="icon-button move-button" data-action="up" aria-label="Move level ${index} up" ${index === 0 ? "disabled" : ""}>↑</button><button type="button" class="icon-button move-button" data-action="down" aria-label="Move level ${index} down" ${index === field.criteria.length - 1 ? "disabled" : ""}>↓</button>` : ""}
        ${!isBoolean ? `<button type="button" class="icon-button" data-action="remove-option" aria-label="Remove ${isGrade ? "level" : "option"} ${isGrade ? index : index + 1} from ${escapeHTML(field.name)}" ${field.criteria.length <= minimum ? "disabled" : ""}>×</button>` : ""}
      </div>`).join("")}
      ${isGrade || isBoolean ? "</details>" : ""}
      ${!isBoolean ? `<button type="button" class="text-button add-option" data-action="add-option" ${field.criteria.length >= (isGrade ? 10 : 255) ? "disabled" : ""}>+ Add ${isGrade ? "grade level" : "option"}</button>` : ""}
      <p class="field-type-note">${typeNotes[field.type]}</p></div></article>`;
  }).join("");
  updateCount();
}

$("#fields").addEventListener("input", (event) => {
  const card = event.target.closest(".field");
  if (!card) return;
  const field = fields.find((item) => item.id === card.dataset.id);
  if (event.target.dataset.option) {
    const index = Number(event.target.closest(".criterion").dataset.index);
    if (event.target.dataset.option === "name" && field.criteria[index].description === field.criteria[index].name) field.criteria[index].description = "";
    field.criteria[index][event.target.dataset.option] = event.target.value;
  } else if (event.target.dataset.prop && !["type", "levelCount"].includes(event.target.dataset.prop)) {
    field[event.target.dataset.prop] = event.target.value;
  }
  markChanged();
});

$("#fields").addEventListener("change", (event) => {
  if (event.target.dataset.prop === "levelCount") {
    const field = fields.find((item) => item.id === event.target.closest(".field").dataset.id);
    const count = Number(event.target.value);
    field.criteria = Array.from({length: count}, (_, index) => field.criteria[index] || {name: String(index), description: ""});
    renderFields(); markChanged(); return;
  }
  if (event.target.dataset.prop !== "type") return;
  const field = fields.find((item) => item.id === event.target.closest(".field").dataset.id);
  const type = event.target.value;
  field.type = type;
  if (type === "score") field.criteria = Array.from({length: 5}, (_, index) => ({name: String(index), description: ""}));
  if (type === "noul") field.criteria = [{name: "true", description: "The proposition is true"}, {name: "false", description: "The proposition is false"}];
  if (type !== "independent" && field.criteria.length < 2) field.criteria.push({name: "other", description: ""});
  renderFields();
  markChanged();
});

$("#fields").addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const field = fields.find((item) => item.id === button.closest(".field").dataset.id);
  const index = Number(button.closest(".criterion")?.dataset.index);
  switch (button.dataset.action) {
    case "remove-field": fields = fields.filter((item) => item !== field); break;
    case "add-option": field.criteria.push({name: `option_${field.criteria.length + 1}`, description: ""}); break;
    case "remove-option": field.criteria.splice(index, 1); break;
    case "up": [field.criteria[index - 1], field.criteria[index]] = [field.criteria[index], field.criteria[index - 1]]; break;
    case "down": [field.criteria[index + 1], field.criteria[index]] = [field.criteria[index], field.criteria[index + 1]]; break;
  }
  renderFields();
  markChanged();
});

$("#add-field").addEventListener("click", () => {
  if (fields.length >= 32) return showError("The playground supports up to 32 output fields.");
  let number = fields.length + 1;
  while (fields.some((field) => field.name === `field_${number}`)) number++;
  fields.push(...fieldsFromSchema({[`field_${number}`]: presets.blank.questions.answer}));
  renderFields();
  markChanged();
  $("#fields .field:last-child input").focus();
});

let presetRequest = 0;
async function loadPreset(name) {
  const request = ++presetRequest;
  const textDemos = ["support_matrix", "inbox_matrix", "ticket_flags", "catalog_choices"];
  if (textDemos.includes(name) && !Object.hasOwn(presets, name)) {
    try {
      const response = await fetch("/static/demo-presets.json");
      if (!response.ok) throw new Error("Could not load the examples. Refresh the page and try again.");
      Object.assign(presets, await response.json());
    } catch (error) { if (request === presetRequest) showError(error.message); return; }
  }
  if (request !== presetRequest) return;
  const preset = presets[name];
  if (textDemos.includes(name)) {
    attachments.forEach((item) => URL.revokeObjectURL(item.url));
    attachments = [];
    renderAttachments();
  }
  $("#state-text").value = preset.text;
  $("#instructions").value = preset.instructions;
  fields = fieldsFromSchema(preset.questions);
  renderFields();
  showError("");
  markChanged();
}
$("#preset").addEventListener("change", (event) => loadPreset(event.target.value));
$("#state-text").addEventListener("input", markChanged);
$("#instructions").addEventListener("input", markChanged);

function mediaKind(file) {
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("audio/")) return "audio";
  if (file.type.startsWith("video/")) return "video";
  const extension = file.name.split(".").pop().toLowerCase();
  if (["jpg", "jpeg", "png", "webp", "bmp"].includes(extension)) return "image";
  if (["wav", "mp3", "m4a", "aac", "ogg", "flac"].includes(extension)) return "audio";
  if (["mp4", "mov", "webm", "mkv", "m4v"].includes(extension)) return "video";
  throw new Error(`Unsupported file: ${file.name}. Add an image, audio clip, or video.`);
}

function addFiles(files) {
  if (busy) return;
  try {
    const pending = Array.from(files).map((file) => ({file, kind: mediaKind(file)}));
    const all = [...attachments, ...pending];
    if (all.reduce((sum, item) => sum + item.file.size, 0) > 200 * 1024 * 1024) throw new Error("Attachments exceed the 200 MB total limit.");
    if (all.filter((item) => item.kind === "image").length > 8 || all.filter((item) => item.kind === "audio").length > 1 || all.filter((item) => item.kind === "video").length > 1) throw new Error("Attach up to 8 images, one audio clip, and one video. Remove an attachment before replacing it.");
    attachments.push(...pending.map((item) => ({...item, id: crypto.randomUUID(), url: URL.createObjectURL(item.file)})));
    renderAttachments();
    markChanged();
    showError("");
  } catch (error) { showError(error.message); }
}

function renderAttachments() {
  $("#attachments").innerHTML = attachments.map((item) => `<div class="attachment" data-id="${item.id}">
    ${item.kind === "image" ? `<img src="${item.url}" alt="Preview of ${escapeHTML(item.file.name)}">` : item.kind === "video" ? `<video src="${item.url}" controls preload="metadata" aria-label="Preview of ${escapeHTML(item.file.name)}"></video>` : ""}
    <div class="attachment-info"><div class="attachment-name">${escapeHTML(item.file.name)}</div><span class="hint">${item.kind === "audio" ? "Speech / audio" : item.kind} · ${(item.file.size / 1024 / 1024).toFixed(2)} MB</span>${item.kind === "audio" ? `<audio src="${item.url}" controls preload="metadata" aria-label="Preview of ${escapeHTML(item.file.name)}"></audio>` : ""}</div>
    <button type="button" class="icon-button" data-remove="${item.id}" aria-label="Remove ${escapeHTML(item.file.name)}">×</button></div>`).join("");
}
$("#media-files").addEventListener("change", (event) => { addFiles(event.target.files); event.target.value = ""; });
$("#attachments").addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove]");
  if (!button) return;
  const item = attachments.find((entry) => entry.id === button.dataset.remove);
  URL.revokeObjectURL(item.url);
  attachments = attachments.filter((entry) => entry !== item);
  renderAttachments();
  markChanged();
});
for (const name of ["dragenter", "dragover"]) $("#dropzone").addEventListener(name, (event) => { event.preventDefault(); if (!busy) $("#dropzone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("#dropzone").addEventListener(name, (event) => { event.preventDefault(); $("#dropzone").classList.remove("dragging"); });
$("#dropzone").addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

function stopRecording() {
  if (recorder?.state === "recording") recorder.stop();
}
$("#record-button").addEventListener("click", async () => {
  if (recorder?.state === "recording") return stopRecording();
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) return showError("Recording is unavailable in this browser. Upload an audio file instead.");
  if (attachments.some((item) => item.kind === "audio")) return showError("Remove the existing audio attachment before recording another clip.");
  $("#record-button").disabled = true;
  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({audio: true});
    const preferred = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"].find((mime) => MediaRecorder.isTypeSupported(mime));
    recorder = new MediaRecorder(recordingStream, preferred ? {mimeType: preferred} : undefined);
    const chunks = [];
    const started = Date.now();
    recorder.addEventListener("dataavailable", (event) => { if (event.data.size) chunks.push(event.data); });
    recorder.addEventListener("stop", () => {
      clearInterval(recordingTimer);
      recordingStream.getTracks().forEach((track) => track.stop());
      const type = recorder.mimeType || "audio/webm";
      const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      addFiles([new File(chunks, `recording.${extension}`, {type})]);
      recorder = null;
      $("#record-button").classList.remove("recording");
      $("#record-label").textContent = "Record speech";
      $("#record-status").textContent = "Recording attached. Preview it before running.";
      syncRunButtons();
    });
    recorder.start();
    $("#record-button").classList.add("recording");
    $("#record-label").textContent = "Stop recording";
    syncRunButtons();
    $("#record-status").textContent = "Recording… 0 / 30 seconds";
    recordingTimer = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      $("#record-status").textContent = `Recording… ${seconds} / 30 seconds`;
      if (seconds >= 29) stopRecording();
    }, 250);
  } catch (error) {
    recordingStream?.getTracks().forEach((track) => track.stop());
    showError(error.name === "NotAllowedError" ? "Microphone access was denied. Allow it in your browser or upload an audio file." : `Could not start recording: ${error.message}`);
  } finally { $("#record-button").disabled = false; }
});
window.addEventListener("beforeunload", () => recordingStream?.getTracks().forEach((track) => track.stop()));

function syncRunButtons() {
  const disabled = !modelReady || busy || recorder?.state === "recording";
  $("#run-button").disabled = disabled;
  $("#compare-button").disabled = disabled;
}

function setBusy(value, comparing = false) {
  busy = value;
  $("#request-form").setAttribute("aria-busy", String(value));
  $("#request-form").querySelectorAll("input, textarea, select, button").forEach((element) => {
    if (value) { element.dataset.previouslyDisabled = String(element.disabled); element.disabled = true; }
    else { element.disabled = element.dataset.previouslyDisabled === "true"; }
  });
  $("#preset").disabled = value;
  syncRunButtons();
  $("#run-label").textContent = value ? "Running…" : "Run all fields";
  $("#compare-button").textContent = value && comparing ? "Comparing…" : "Compare with Gemma";
  $("#run-status span:last-child").textContent = comparing ? "Running the parallel scorer, then normal Gemma generation…" : "Processing input and scoring all fields…";
  $("#run-status").hidden = !value;
  if (value) { $("#result-badge").textContent = "Running"; $("#result-badge").className = "result-badge"; }
  else if (modelReady) { $("#model-status").textContent = "Gemma 4 E2B · ready"; $("#status-dot").className = "status-dot"; }
}

$("#request-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !modelReady || recorder?.state === "recording") return;
  const comparing = event.submitter?.id === "compare-button";
  try {
    const questions = buildSchema();
    if (!$("#state-text").value.trim() && attachments.length === 0) throw new Error("Add some text or a media attachment first.");
    const spec = {text: $("#state-text").value, instructions: $("#instructions").value, questions, media: attachments.map((item) => ({name: item.file.name, kind: item.kind}))};
    const body = new FormData();
    body.append("spec", JSON.stringify(spec));
    attachments.forEach((item) => body.append("media", item.file));
    showError("");
    setBusy(true, comparing);
    const started = performance.now();
    const response = await fetch(comparing ? "/api/compare" : "/api/run", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || "The request failed.");
    data.media.forEach((media, index) => { attachments[index].kind = media.kind; });
    renderAttachments();
    result = {request: spec, response: {...data, browser_round_trip_seconds: (performance.now() - started) / 1000}};
    activeView = comparing ? "compare" : "answers";
    renderResults();
    if (window.innerWidth <= 720) $(".results-column").scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start"});
  } catch (error) {
    showError(error.message === "Failed to fetch" ? "The local server is unreachable. Restart it, then try again." : error.message);
    $("#result-badge").textContent = result ? "Previous result" : "Run failed";
    $("#result-badge").className = "result-badge stale";
  } finally { if (busy) setBusy(false); }
});

function percentText(value) {
  const percent = Math.max(0, Math.min(100, value * 100));
  return `${percent > 0 && percent < .1 ? "<0.1" : percent > 99.9 && percent < 100 ? ">99.9" : percent.toFixed(1)}%`;
}

function probabilityRow(label, value, winner) {
  const percent = Math.max(0, Math.min(100, value * 100));
  return `<div class="prob-row ${winner ? "winner" : ""}"><div class="prob-heading"><span class="prob-label">${escapeHTML(label)}</span><span class="prob-value">${escapeHTML(percentText(value))}</span></div><div class="prob-track" aria-hidden="true"><div class="prob-fill" style="width:${percent}%"></div></div></div>`;
}

function renderComparison(comparison) {
  $("#view-compare").hidden = !comparison;
  if (!comparison) { $("#compare-view").innerHTML = ""; return; }
  const ratio = comparison.normal_over_parallel;
  const verdict = ratio === null ? "Normal Gemma did not return a valid complete answer" : ratio >= 1 ? `${ratio.toFixed(2)}× faster this run` : `${(1 / ratio).toFixed(2)}× slower this run`;
  const valueText = (value) => typeof value === "object" ? JSON.stringify(value) : String(value);
  $("#compare-view").innerHTML = `<div class="comparison-times">
    <div><span>Parallel scorer</span><strong>${Math.round(comparison.seconds.parallel * 1000).toLocaleString()} <small>ms</small></strong><span>Direct probabilities</span></div>
    <div><span>Normal Gemma</span><strong>${Math.round(comparison.seconds.normal * 1000).toLocaleString()} <small>ms</small></strong><span>${comparison.normal.output_tokens} generated tokens</span></div>
    </div><p class="comparison-verdict">${escapeHTML(verdict)}</p>
    <p class="comparison-note">One run each · same model and input · model loading excluded</p>
    ${comparison.normal.error ? `<p class="error">${escapeHTML(comparison.normal.error)}</p>` : ""}
    <div class="comparison-table-wrap"><table class="comparison-table"><thead><tr><th scope="col">Field</th><th scope="col">Parallel</th><th scope="col">Normal Gemma</th></tr></thead><tbody>${Object.entries(comparison.parallel_values).map(([name, value]) => `<tr><th scope="row">${escapeHTML(name)}<span class="comparison-match ${comparison.agreement[name] === false ? "different" : ""}">${comparison.agreement[name] === null ? "Invalid output" : comparison.agreement[name] ? "Same answer" : "Different answers"}</span></th><td>${escapeHTML(valueText(value))}</td><td>${comparison.normal.answers && Object.hasOwn(comparison.normal.answers, name) ? escapeHTML(valueText(comparison.normal.answers[name])) : "—"}</td></tr>`).join("")}</tbody></table></div>
    <p class="comparison-note">Grades compare the most likely level; yes/no uses a 50% threshold. See Answers for the scorer’s probabilities and expected grades. Agreement does not establish accuracy.</p>
    <details class="comparison-details"><summary>Normal Gemma’s generated answer</summary><pre>${escapeHTML(comparison.normal.raw_text)}</pre></details>
    <details class="comparison-details"><summary>What these timings include</summary>${Object.values(comparison.methodology).map((text) => `<p>${escapeHTML(text)}</p>`).join("")}</details>`;
}

function renderResults() {
  const data = result.response;
  $("#empty-results").hidden = true;
  $("#timings").hidden = false;
  $("#result-footer").hidden = false;
  $("#export-result").disabled = false;
  $("#result-badge").textContent = "Complete";
  $("#result-badge").className = "result-badge complete";
  const execution = data.execution;
  const stages = [(data.media_prepare_seconds || 0) + (execution.preprocess_seconds || 0), execution.prefill_seconds || 0, execution.branch_seconds || 0];
  const stageSum = stages.reduce((sum, value) => sum + value, 0) || 1;
  const batches = [...(execution.branch_batch_sizes || []), ...(execution.candidate_batch_sizes || [])];
  const decisions = execution.primitive_fields ?? batches.reduce((sum, count) => sum + count, 0);
  const timingSeconds = data.comparison ? data.comparison.seconds.parallel : data.browser_round_trip_seconds;
  $("#timings").innerHTML = `<div class="timing-head"><div class="timing-number">${Math.round(timingSeconds * 1000).toLocaleString()}<span>ms ${data.comparison ? "parallel scorer" : "total"}</span></div><div class="timing-caption">${decisions} ${decisions === 1 ? "decision" : "decisions"} · ${batches.length} GPU ${batches.length === 1 ? "batch" : "batches"}</div></div><div class="timing-bar" aria-hidden="true">${stages.map((value) => `<i style="width:${value / stageSum * 100}%"></i>`).join("")}</div><div class="timing-legend">${stages.map((value, index) => `<span>${["Prepare input", "Shared state", "Score fields"][index]} <b>${Math.round(value * 1000)} ms</b></span>`).join("")}</div>`;
  renderComparison(data.comparison);
  $("#answers-view").innerHTML = Object.entries(data.answers).map(([name, answer]) => {
    const definition = result.request.questions[name];
    let headline = "", subtitle = "", probabilities = answer.probabilities;
    if (answer.type === "choice") {
      headline = escapeHTML(answer.choice);
      subtitle = `${escapeHTML(percentText(answer.selected_probability))} probability of the selected option`;
    } else if (answer.type === "score") {
      headline = `${answer.score.toFixed(2)} <small>/ ${Object.keys(answer.legend).length - 1}</small>`;
      subtitle = "Expected grade · levels start at 0";
    } else if (answer.type === "noul") {
      headline = `${escapeHTML(percentText(answer.noul))} <small>yes</small>`;
      subtitle = "Probability that the proposition is true";
      probabilities = {Yes: answer.noul, No: 1 - answer.noul};
    } else { subtitle = "Independent yes probabilities · multiple labels can be true"; }
    const winner = Object.keys(probabilities).reduce((best, key) => probabilities[key] > probabilities[best] ? key : best);
    return `<article class="answer-card"><div class="answer-heading"><span class="answer-key">${escapeHTML(name)}</span><span class="type-tag">${typeNames[answer.type]}</span></div>${headline ? `<div class="answer-value">${headline}</div>` : ""}<div class="answer-subtitle">${subtitle}</div>${Object.entries(probabilities).map(([key, value]) => {
      const label = answer.type === "score" ? `${key} · ${answer.legend[key]}` : key;
      return probabilityRow(label, value, answer.type === "independent" ? value >= .5 : key === winner);
    }).join("")}<details class="input-limits"><summary>Question & criteria</summary><p>${escapeHTML(definition.instructions)}</p>${answer.type !== "score" ? `<p>${Object.entries(definition.criteria || {}).map(([key, description]) => `${escapeHTML(key)}: ${escapeHTML(description)}`).join("<br>")}</p>` : ""}</details></article>`;
  }).join("");
  $("#result-json").textContent = JSON.stringify(result, null, 2);
  selectView(activeView);
}

function selectView(view) {
  activeView = view;
  for (const name of ["answers", "compare", "json"]) {
    $(`#view-${name}`).setAttribute("aria-selected", String(name === view));
    $(`#${name}-view`).hidden = name !== view;
  }
}
$("#view-answers").addEventListener("click", () => selectView("answers"));
$("#view-json").addEventListener("click", () => selectView("json"));
$("#view-compare").addEventListener("click", () => selectView("compare"));
$("#copy-result").addEventListener("click", async () => {
  if (!result) return;
  try { await navigator.clipboard.writeText(JSON.stringify(result, null, 2)); $("#copy-result").textContent = "Copied"; setTimeout(() => { $("#copy-result").textContent = "Copy JSON"; }, 1600); }
  catch { showError("Clipboard access is unavailable. Use Export to save the result."); }
});
$("#export-result").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = "decision-result.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("#edit-json").addEventListener("click", () => {
  try { $("#schema-json").value = JSON.stringify(buildSchema(), null, 2); }
  catch { $("#schema-json").value = JSON.stringify(buildSchema(false), null, 2); }
  $("#schema-error").hidden = true;
  $("#schema-dialog").showModal();
});
$("#apply-schema").addEventListener("click", async () => {
  $("#apply-schema").disabled = true;
  try {
    const response = await fetch("/api/validate", {method: "POST", headers: {"Content-Type": "application/json"}, body: `{"questions":${$("#schema-json").value}}`});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Invalid schema");
    fields = fieldsFromSchema(data.questions);
    renderFields(); markChanged(); $("#schema-dialog").close();
  } catch (error) { $("#schema-error").textContent = error.message; $("#schema-error").hidden = false; }
  finally { $("#apply-schema").disabled = false; }
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !$("#schema-dialog").open) {
    event.preventDefault(); $("#request-form").requestSubmit($("#run-button"));
  }
});

async function pollStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Status unavailable");
    const status = await response.json();
    modelReady = status.ready && (!status.busy || busy);
    $("#model-status").textContent = status.error ? "Model unavailable" : status.busy ? "Gemma 4 E2B · running" : status.ready ? "Gemma 4 E2B · ready" : "Loading Gemma 4 E2B…";
    $("#status-dot").className = `status-dot${status.error ? " offline" : !status.ready || status.busy ? " loading" : ""}`;
    if (status.error) showError(status.error);
  } catch {
    modelReady = false;
    $("#model-status").textContent = "Server offline";
    $("#status-dot").className = "status-dot offline";
  }
  syncRunButtons();
  setTimeout(pollStatus, modelReady ? 5000 : 1500);
}

try {
  const draft = JSON.parse(localStorage.getItem("decision-lab-draft-v1"));
  if (!draft || !Array.isArray(draft.fields) || !draft.fields.length) throw new Error("No draft");
  fields = draft.fields;
  $("#state-text").value = draft.text || "";
  $("#instructions").value = draft.instructions || "";
  renderFields();
} catch { loadPreset("animals"); }
pollStatus();
