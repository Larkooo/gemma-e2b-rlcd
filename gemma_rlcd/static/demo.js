"use strict";
const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
let config, media, mediaURL, active = false, ready = false, controller, finalResult = null;
let checks = [], rows = new Map(), completed = {parallel: new Map(), normal: new Map()};
let clocks = {}, rawText = "", parallelLines = [], events = [], firstAnswer = {}, frame = null, sampleVersion = 0;

function error(message) { $("#error").textContent = message; $("#error").hidden = !message; }
function seconds(value) { return `${value.toFixed(2)}<span>s</span>`; }
function count() { return Number($("#output-count").value); }
function setMedia(file, credit = "") {
  if (active) return;
  if (file.size > 200 * 1024 * 1024) return error("The media file exceeds 200 MB.");
  const extension = file.name.split(".").pop().toLowerCase();
  const kind = file.type.startsWith("image/") || ["jpg","jpeg","png","webp","bmp"].includes(extension) ? "image" : file.type.startsWith("video/") || ["mp4","mov","webm","mkv","m4v"].includes(extension) ? "video" : null;
  if (!kind) return error("Choose an image or video file.");
  if (mediaURL) URL.revokeObjectURL(mediaURL);
  media = {file, kind}; mediaURL = URL.createObjectURL(file);
  $("#image-preview").hidden = kind !== "image";
  $("#video-preview").hidden = kind !== "video";
  $("#image-preview").removeAttribute("src");
  $("#video-preview").removeAttribute("src");
  $(`#${kind}-preview`).src = mediaURL;
  $("#image-preview").alt = file.name;
  $("#file-name").textContent = file.name;
  $("#credit").innerHTML = credit;
  reset(); error("");
}
async function sample() {
  const version = ++sampleVersion;
  $("#sample").disabled = true;
  try {
    const response = await fetch(config.sample.url);
    if (!response.ok) throw new Error("Could not load the sample photo.");
    const blob = await response.blob();
    if (version !== sampleVersion || active) return;
    setMedia(new File([blob], "times-square.jpg", {type:"image/jpeg"}), `<a href="${escapeHTML(config.sample.source)}" target="_blank" rel="noreferrer">${escapeHTML(config.sample.author)} · ${escapeHTML(config.sample.license)}</a>`);
  } catch (failure) { error(failure.message); }
  finally { $("#sample").disabled = active; }
}
function reset() {
  finalResult = null; rawText = ""; parallelLines = []; events = []; firstAnswer = {}; clocks = {};
  completed = {parallel: new Map(), normal: new Map()};
  $("#headline-count").textContent = count();
  $("#parallel-stream").textContent = "Answers appear as soon as each batch completes.";
  $("#normal-stream").textContent = "Real token output will stream here.";
  $("#token-count").textContent = "0 tokens";
  $("#run-phase").textContent = "Ready";
  $("#agreement").textContent = "";
  $("#verdict").textContent = "Upload a scene or try the sample. Results are measured live.";
  $("#export").disabled = true;
  for (const method of ["parallel","normal"]) {
    $(`#${method}-clock`).innerHTML = seconds(0);
    $(`#${method}-state`).textContent = "Waiting";
    $(`#${method}-first`).textContent = "First answer —";
    $(`#${method}-progress`).style.width = "0%";
    $(`#${method}-count`).textContent = `0 / ${count()} decisions`;
  }
  renderGrid(); syncButtons();
}
function renderGrid() {
  checks = [];
  $("#output-grid").innerHTML = config.groups.map((group) => {
    const items = group.checks.slice(0, count() / config.groups.length);
    return `<section class="group"><h3 class="group-title">${escapeHTML(group.name)} · ${items.length} checks</h3><div class="check-grid">${items.map((check) => {
      const path = `${group.id}.${check.id}`; checks.push(path);
      return `<div class="check" data-path="${escapeHTML(path)}" title="${escapeHTML(check.description)}"><span class="check-name">${escapeHTML(check.label)}</span><div class="values"><span>P <b data-method="parallel">—</b></span><span>G <b data-method="normal">—</b></span></div></div>`;
    }).join("")}</div></section>`;
  }).join("");
  rows = new Map(Array.from(document.querySelectorAll(".check"), (row) => [row.dataset.path, row]));
  filter();
}
function updateDecision(method, path, value, probability, elapsed) {
  const row = rows.get(path);
  if (!row || completed[method].has(path) && completed[method].get(path).value === value) return;
  completed[method].set(path, {value, probability});
  const cell = row.querySelector(`[data-method="${method}"]`);
  cell.textContent = method === "parallel" ? `${Math.round(probability * 100)}%` : value ? "yes" : "no";
  cell.className = value ? "yes" : "no";
  row.classList.add("flash"); setTimeout(() => row.classList.remove("flash"), 600);
  const parallel = completed.parallel.get(path), normal = completed.normal.get(path);
  row.classList.toggle("detected", Boolean(parallel?.value || normal?.value));
  row.classList.toggle("different", Boolean(parallel && normal && parallel.value !== normal.value));
  if (firstAnswer[method] === undefined) {
    firstAnswer[method] = elapsed;
    $(`#${method}-first`).textContent = `First answer ${elapsed.toFixed(2)} s`;
  }
  $(`#${method}-count`).textContent = `${completed[method].size} / ${count()} decisions`;
  $(`#${method}-progress`).style.width = `${100 * completed[method].size / count()}%`;
  filter();
}
function filter() {
  const mode = $("#filter").value;
  let visible = 0;
  for (const [path, row] of rows) {
    const p = completed.parallel.get(path), n = completed.normal.get(path);
    row.hidden = mode === "yes" ? !(p?.value || n?.value) : mode === "different" ? !(p && n && p.value !== n.value) : false;
    if (!row.hidden) visible++;
  }
  for (const group of document.querySelectorAll(".group")) group.hidden = !Array.from(group.querySelectorAll(".check")).some((row) => !row.hidden);
  $("#no-matches").hidden = visible > 0;
}
function tick() {
  for (const [method, clock] of Object.entries(clocks)) {
    if (clock.running) $(`#${method}-clock`).innerHTML = seconds((performance.now() - clock.start) / 1000);
  }
  if (active) frame = requestAnimationFrame(tick);
}
function handle(event) {
  events.push(event.type === "complete" ? {type: "complete"} : event);
  const method = event.method;
  if (event.type === "accepted") { $("#run-phase").textContent = "Preparing media"; return; }
  if (event.type === "error") throw new Error(event.error);
  if (event.type === "race_start") {
    const start = performance.now() - event.media_seconds * 1000;
    for (const name of ["parallel","normal"]) {
      clocks[name] = {running:true, start};
      $(`#${name}-state`).textContent = "Running";
    }
    $("#run-phase").textContent = "Both running · live";
    $("#verdict").textContent = "Both paths started together. Watch the answers arrive…";
  } else if (event.type === "phase_start") {
    if (!clocks[method]) clocks[method] = {running:true, start:performance.now() - event.media_seconds * 1000};
    $(`#${method}-state`).textContent = "Running";
  } else if (event.type === "answer") {
    const path = event.path.join(".");
    const probability = event.answer.probabilities.yes;
    updateDecision("parallel",path,event.value,probability,event.seconds);
    parallelLines.push(`${path}: ${event.value}  (${(probability*100).toFixed(1)}% yes)`);
    $("#parallel-stream").textContent = parallelLines.join("\n");
    $("#parallel-stream").scrollTop = $("#parallel-stream").scrollHeight;
  } else if (event.type === "token") {
    rawText += event.text;
    $("#normal-stream").textContent = rawText;
    $("#normal-stream").scrollTop = $("#normal-stream").scrollHeight;
    $("#token-count").textContent = `${event.tokens} tokens`;
    for (const [path,value] of Object.entries(VisualDemo.partialBooleans(rawText))) updateDecision("normal",path,value,null,event.seconds);
  } else if (event.type === "phase_complete") {
    clocks[method].running = false;
    $(`#${method}-clock`).innerHTML = seconds(event.seconds);
    $(`#${method}-state`).textContent = event.valid ? "Complete" : "Invalid output";
    const other = method === "parallel" ? "normal" : "parallel";
    if (clocks[other]?.running) {
      const name = method === "parallel" ? "Parallel scorer" : "Normal Gemma";
      const otherName = other === "parallel" ? "Parallel scorer" : "Normal Gemma";
      $("#run-phase").textContent = `${otherName} still running`;
      $("#verdict").textContent = `${name} ${event.valid ? "finished" : "returned invalid output"} in ${event.seconds.toFixed(2)} s. ${otherName} is still working…`;
    }
  } else if (event.type === "complete") {
    finalResult = event.result;
    const comparison = finalResult.comparison;
    let matched = 0;
    if (comparison.normal.valid) {
      for (const [group, values] of Object.entries(comparison.normal.answers)) {
        for (const [key,value] of Object.entries(values)) {
          const path = `${group}.${key}`;
          updateDecision("normal",path,value,null,comparison.seconds.normal);
          if (completed.parallel.get(path)?.value === value) matched++;
        }
      }
      $("#agreement").textContent = `${matched} / ${count()} matched`;
      const ratio = comparison.normal_over_parallel;
      $("#verdict").innerHTML = `<strong>${ratio >= 1 ? ratio.toFixed(2) : (1 / ratio).toFixed(2)}× ${ratio >= 1 ? "faster" : "slower"}</strong> this run · ${matched} / ${count()} matching answers`;
    } else {
      $("#verdict").textContent = "Normal Gemma returned an invalid answer. No valid-response speedup is reported.";
      error(comparison.normal.error);
      $("#agreement").textContent = "Normal output invalid";
    }
    $("#run-phase").textContent = "Comparison complete";
    $("#export").disabled = false;
  }
}
function syncButtons() {
  $("#run").disabled = !ready || active || !media;
  $("#stop").hidden = !active;
  for (const selector of ["#sample","#media-file","#output-count","#instructions"]) $(selector).disabled = active;
  $("#run").textContent = active ? "Streaming…" : "Compare · stream answers";
}
async function run() {
  if (active || !ready || !media) return;
  reset(); error(""); active = true; controller = new AbortController(); syncButtons(); tick();
  const spec = {text:"", instructions:$("#instructions").value, questions:VisualDemo.questions(config,count()), media:[{name:media.file.name,kind:media.kind}]};
  const form = new FormData(); form.append("spec",JSON.stringify(spec)); form.append("media",media.file);
  try {
    const response = await fetch("/api/compare-stream",{method:"POST",body:form,signal:controller.signal});
    if (!response.ok) { const data=await response.json(); throw new Error(data.error || "Comparison failed."); }
    const reader=response.body.getReader(), decoder=new TextDecoder(); let pending="";
    while (true) {
      const {value,done}=await reader.read(); pending += decoder.decode(value,{stream:!done});
      let end;
      while ((end=pending.indexOf("\n"))>=0) { const line=pending.slice(0,end); pending=pending.slice(end+1); if(line.trim()) handle(JSON.parse(line)); }
      if(done) break;
    }
    if(pending.trim()) handle(JSON.parse(pending));
    if(!finalResult) throw new Error("The stream ended before the comparison completed.");
    finalResult = {request:spec,response:finalResult,stream_events:events,first_answer_seconds:firstAnswer};
    $("#run-note").textContent = "Complete. Export preserves the answers, events, and measured timings.";
  } catch(failure) {
    controller.abort();
    const cancelled=failure.name==="AbortError";
    if(!cancelled) error(failure.message);
    $("#run-phase").textContent=cancelled ? "Stopped" : "Run failed";
    $("#verdict").textContent=cancelled ? "Stopped. Partial results are shown; no complete-response comparison." : "The comparison did not complete.";
    for(const [method,clock] of Object.entries(clocks)) if(clock.running) $(`#${method}-state`).textContent=cancelled ? "Stopped" : "Failed";
  } finally {
    active=false; cancelAnimationFrame(frame); for(const clock of Object.values(clocks)) clock.running=false;
    syncButtons(); status();
  }
}
async function status() {
  try {
    const response=await fetch("/api/status"), state=await response.json();
    ready=state.ready && !state.busy;
    $("#model-status").textContent=state.error ? "Model unavailable" : state.busy ? "Gemma · running" : state.ready ? "Gemma 4 E2B · ready" : "Loading model…";
    if(state.error) error(state.error);
  } catch { ready=false; $("#model-status").textContent="Server disconnected"; }
  syncButtons();
}
$("#run").addEventListener("click",run);
$("#stop").addEventListener("click",()=>controller?.abort());
$("#sample").addEventListener("click",sample);
$("#filter").addEventListener("change",filter);
$("#output-count").addEventListener("change",reset);
$("#instructions").addEventListener("input",()=>{ if(finalResult) $("#run-note").textContent="Instructions changed. Run again to update the results."; });
$("#media-file").addEventListener("change",(event)=>{ sampleVersion++; if(event.target.files[0]) setMedia(event.target.files[0]); event.target.value=""; });
$("#dropzone").addEventListener("click",(event)=>{ if(event.target.tagName!=="VIDEO" && !active) $("#media-file").click(); });
$("#dropzone").addEventListener("keydown",(event)=>{ if(["Enter"," "].includes(event.key) && !active) {event.preventDefault(); $("#media-file").click();} });
for(const name of ["dragover","dragenter"]) $("#dropzone").addEventListener(name,(event)=>{event.preventDefault(); if(!active) $("#dropzone").classList.add("dragging");});
for(const name of ["dragleave","drop"]) $("#dropzone").addEventListener(name,(event)=>{event.preventDefault(); $("#dropzone").classList.remove("dragging");});
$("#dropzone").addEventListener("drop",(event)=>{sampleVersion++; if(event.dataTransfer.files[0]) setMedia(event.dataTransfer.files[0]);});
$("#export").addEventListener("click",()=>{
  if(!finalResult) return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(finalResult,null,2)],{type:"application/json"}));
  const link=document.createElement("a"); link.href=url; link.download="gemma-visual-comparison.json"; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function init() {
  try {
    const response=await fetch("/static/visual-demo.json"); if(!response.ok) throw new Error("Could not load the visual checks.");
    config=await response.json(); $("#instructions").value=config.instructions; reset(); await sample(); await status(); setInterval(status,2000);
  } catch(failure) { error(failure.message); }
}
init();
