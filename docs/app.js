"use strict";

// Every frame and heatmap comes from Figure 2, displayed directly from the
// unmodified research image. The coordinates exclude its labels and gutters.
const figure = {
  width: 2530,
  height: 752,
  left: 131,
  columnWidth: 240,
  rows: { frame: [33, 179], attention: [394, 179], combined: [574, 178] },
};
const slider = document.querySelector("#burst-slider");
const scene = document.querySelector("#scene-image");
const map = document.querySelector("#map-image");
const playButton = document.querySelector("#play-bursts");
const layerButtons = [...document.querySelectorAll("[data-layer]")];
let currentLayer = "combined";
let playback = null;

function imageViewport(burst, layer) {
  const [y, height] = figure.rows[layer];
  const x = figure.left + burst * figure.columnWidth;
  return `${x} ${y} ${Math.min(figure.columnWidth, figure.width - x)} ${height}`;
}

function updateExplorer() {
  const burst = Number(slider.value);
  const initialPrior = burst === 0 && currentLayer === "combined";
  scene.setAttribute("viewBox", imageViewport(burst, "frame"));
  map.setAttribute("viewBox", imageViewport(burst, currentLayer));
  map.style.opacity = currentLayer === "frame" || initialPrior ? "0" : "0.9";
  document.querySelector("#scene-title").textContent = `Burst ${burst}: original frame from the robotic teleoperation sequence`;
  document.querySelector("#map-title").textContent = initialPrior
    ? "Burst 0 has no combined spatial prior; the uncalibrated scene is displayed"
    : `Burst ${burst}: ${currentLayer === "attention" ? "decoder attention" : currentLayer === "combined" ? "combined spatial focus prior" : "original scene"}`;
  const formatted = String(burst).padStart(2, "0");
  document.querySelector("#burst-number").textContent = formatted;
  document.querySelector("#burst-output").value = formatted;
  slider.setAttribute("aria-valuetext", `Burst ${burst} of 9${initialPrior ? ", initial observation without a prior" : ""}`);
  document.querySelector("#explorer-caption").textContent = initialPrior
    ? "The first burst establishes context. No spatial prior exists yet."
    : currentLayer === "frame"
      ? "A representative frame from the paper’s robotic teleoperation sequence."
      : currentLayer === "attention"
        ? "The decoder’s attention after this response becomes the next burst’s prior."
        : "Attention from the previous response meets new visual change.";
  layerButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.layer === currentLayer)));
}

function stopPlayback() {
  clearInterval(playback);
  playback = null;
  playButton.setAttribute("aria-label", "Play burst sequence");
  playButton.setAttribute("aria-pressed", "false");
  playButton.querySelector("path").setAttribute("d", "m7 4 9 6-9 6Z");
}

playButton.addEventListener("click", () => {
  if (playback) return stopPlayback();
  if (Number(slider.value) === 9) slider.value = "0";
  updateExplorer();
  playButton.setAttribute("aria-label", "Pause burst sequence");
  playButton.setAttribute("aria-pressed", "true");
  playButton.querySelector("path").setAttribute("d", "M5 4h3v12H5Zm7 0h3v12h-3Z");
  playback = setInterval(() => {
    slider.value = String(Number(slider.value) + 1);
    updateExplorer();
    if (Number(slider.value) === 9) stopPlayback();
  }, 1500);
});
slider.addEventListener("input", () => { stopPlayback(); updateExplorer(); });
layerButtons.forEach((button) => button.addEventListener("click", () => {
  currentLayer = button.dataset.layer;
  updateExplorer();
}));
document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
updateExplorer();

// This small diagram is schematic, not an additional experiment.
const change = [0.12,0.17,0.23,0.18,0.12,0.19,0.42,0.82,0.74,0.2,0.28,0.85,1,0.88,0.34,0.15,0.43,0.81,0.65,0.3,0.1,0.22,0.3,0.22,0.12];
const attention = [0.12,0.15,0.18,0.2,0.14,0.14,0.19,0.44,0.62,0.28,0.15,0.3,0.82,1,0.48,0.12,0.22,0.54,0.8,0.39,0.1,0.13,0.28,0.38,0.16];
const gridData = { "change-grid": change, "attention-grid": attention, "focus-grid": change.map((value, index) => Math.max(0.08, value * attention[index])) };
for (const [id, values] of Object.entries(gridData)) {
  const grid = document.getElementById(id);
  values.forEach((value) => {
    const cell = document.createElement("i");
    cell.style.setProperty("--strength", value);
    grid.append(cell);
  });
}

// Values are transcribed from Tables 1 and 2 of the supplied camera-ready paper.
// Both FOCUS variants share a call count: gating alone determines the calls.
const benchmarks = {
  egoschema: {
    title: "Accuracy vs. model calls",
    accessibleTitle: "EgoSchema accuracy and VLM calls",
    direction: "↖ HIGHER ACCURACY, FEWER CALLS",
    xLabel: "Mean VLM calls per three-minute clip",
    yLabel: "Accuracy (%)",
    xMax: 60, xTicks: [0, 10, 20, 30, 40, 50, 60],
    yMin: 70, yMax: 76, yTicks: [70, 71, 72, 73, 74, 75, 76],
    context: "The result is efficiency at comparable accuracy, not an accuracy win. Gating determines the call budget; feedback steers spatial focus. Both FOCUS variants make the same number of calls.",
    points: [
      { name: "Uniform", x: 55, y: 74.6, kind: "baseline", dx: -13, dy: -10, anchor: "end" },
      { name: "BOLT", x: 55, y: 74.2, kind: "baseline", dx: -13, dy: 18, anchor: "end" },
      { name: "DyCoke", x: 12, y: 71.4, kind: "baseline", dx: 13, dy: -8 },
      { name: "Gating only", x: 25.4, y: 73.6, kind: "gating", dx: 14, dy: 17 },
      { name: "FOCUS", x: 25.4, y: 74.8, kind: "focus", dx: 14, dy: -13, value: "74.8% · 25.4 calls" },
    ],
  },
  captioning: {
    title: "Caption quality vs. model calls",
    accessibleTitle: "Video caption quality and total VLM calls",
    direction: "↖ HIGHER QUALITY, FEWER CALLS",
    xLabel: "Total VLM calls across 25 videos",
    yLabel: "Quality (1–5)",
    xMax: 560, xTicks: [0, 100, 200, 300, 400, 500],
    yMin: 3.5, yMax: 4.1, yTicks: [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1],
    context: "Quality differences are not statistically significant in this small, LLM-judged study. The 57% reduction is in VLM calls; it does not directly measure total runtime. Captioning decoding settings are not fully matched.",
    points: [
      { name: "Uniform", x: 516, y: 3.81, kind: "baseline", dx: -12, dy: -13, anchor: "end" },
      { name: "BOLT", x: 263, y: 3.91, kind: "baseline", dx: 14, dy: -8 },
      { name: "DyCoke", x: 263, y: 3.81, kind: "baseline", dx: 14, dy: 18 },
      { name: "Gating only", x: 222, y: 3.89, kind: "gating", dx: -14, dy: 18, anchor: "end" },
      { name: "FOCUS", x: 222, y: 3.92, kind: "focus", dx: -14, dy: -26, anchor: "end", value: "3.92 / 5 · 222 calls" },
    ],
  },
};
const chart = document.querySelector("#results-chart");
const svgNamespace = "http://www.w3.org/2000/svg";
function svgElement(name, attributes = {}, text) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function renderChart(name) {
  const data = benchmarks[name];
  const bounds = { left: 64, right: 612, top: 35, bottom: 290 };
  const x = (value) => bounds.left + value / data.xMax * (bounds.right - bounds.left);
  const y = (value) => bounds.bottom - (value - data.yMin) / (data.yMax - data.yMin) * (bounds.bottom - bounds.top);
  chart.replaceChildren();
  chart.append(svgElement("title", { id: "chart-title" }, data.accessibleTitle));
  chart.append(svgElement("desc", { id: "chart-description" }, data.points.map((p) => `${p.name}: ${p.y}${name === "egoschema" ? "% accuracy" : " quality out of 5"}, ${p.x} calls`).join(". ") + ". Exact values are also in the results table."));
  data.yTicks.forEach((tick) => {
    chart.append(svgElement("line", { x1: bounds.left, x2: bounds.right, y1: y(tick), y2: y(tick), class: "chart-grid" }));
    chart.append(svgElement("text", { x: bounds.left - 13, y: y(tick) + 3, "text-anchor": "end", class: "chart-tick" }, name === "captioning" ? tick.toFixed(1) : tick));
  });
  data.xTicks.forEach((tick) => {
    chart.append(svgElement("text", { x: x(tick), y: bounds.bottom + 23, "text-anchor": "middle", class: "chart-tick" }, tick));
  });
  chart.append(svgElement("line", { x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom, stroke: "#aeb9a2", "stroke-width": "1" }));
  chart.append(svgElement("text", { x: (bounds.left + bounds.right) / 2, y: 346, "text-anchor": "middle", class: "chart-axis-label" }, data.xLabel));
  chart.append(svgElement("text", { x: -162, y: 17, transform: "rotate(-90)", "text-anchor": "middle", class: "chart-axis-label" }, data.yLabel));
  const focus = data.points.find((point) => point.kind === "focus");
  const gating = data.points.find((point) => point.kind === "gating");
  chart.append(svgElement("line", { x1: x(focus.x), x2: x(gating.x), y1: y(focus.y), y2: y(gating.y), stroke: "#9cb87d", "stroke-dasharray": "3 4" }));
  data.points.forEach((point) => {
    const group = svgElement("g", { class: "chart-point", tabindex: "0", role: "img", "aria-label": `${point.name}: ${point.y}${name === "egoschema" ? "% accuracy" : " out of 5 quality"}, ${point.x} VLM calls` });
    group.append(svgElement("title", {}, `${point.name}: ${point.y}${name === "egoschema" ? "%" : " / 5"}, ${point.x} calls`));
    if (point.kind === "focus") group.append(svgElement("circle", { cx: x(point.x), cy: y(point.y), r: 14, fill: "#b6db88", opacity: ".28" }));
    group.append(svgElement("circle", { cx: x(point.x), cy: y(point.y), r: point.kind === "focus" ? 6 : 4.5, fill: point.kind === "focus" ? "#709f3c" : point.kind === "gating" ? "#fafcf5" : "#a9b09f", stroke: point.kind === "gating" ? "#709f3c" : "#fafcf5", "stroke-width": point.kind === "gating" ? 1.5 : 2 }));
    group.append(svgElement("text", { x: x(point.x) + point.dx, y: y(point.y) + point.dy, "text-anchor": point.anchor || "start", class: `chart-point-label${point.kind === "focus" ? " emphasized" : ""}` }, point.name));
    if (point.value) group.append(svgElement("text", { x: x(point.x) + point.dx, y: y(point.y) + point.dy + 14, "text-anchor": point.anchor || "start", class: "chart-point-value" }, point.value));
    chart.append(group);
  });
  document.querySelector("#chart-heading").textContent = data.title;
  document.querySelector("#chart-direction").textContent = data.direction;
  document.querySelector("#results-context").textContent = data.context;
}

const benchmarkTabs = [...document.querySelectorAll("[data-benchmark]")];
function selectBenchmark(tab) {
  benchmarkTabs.forEach((item) => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
  });
  renderChart(tab.dataset.benchmark);
}
benchmarkTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectBenchmark(tab));
  tab.addEventListener("keydown", (event) => {
    const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? benchmarkTabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + benchmarkTabs.length) % benchmarkTabs.length;
    benchmarkTabs[nextIndex].focus();
    selectBenchmark(benchmarkTabs[nextIndex]);
  });
});
renderChart("egoschema");

const menuButton = document.querySelector(".menu-toggle");
const nav = document.querySelector("#main-nav");
function closeMenu() {
  nav.classList.remove("is-open");
  menuButton.setAttribute("aria-expanded", "false");
}
menuButton.addEventListener("click", () => {
  const expanded = menuButton.getAttribute("aria-expanded") !== "true";
  nav.classList.toggle("is-open", expanded);
  menuButton.setAttribute("aria-expanded", String(expanded));
});
nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
    closeMenu();
    menuButton.focus();
  }
});
document.addEventListener("click", (event) => { if (!event.target.closest(".site-header")) closeMenu(); });
window.matchMedia("(min-width: 651px)").addEventListener("change", closeMenu);

document.querySelector("#copy-citation").addEventListener("click", async () => {
  const text = document.querySelector("#bibtex").textContent;
  const status = document.querySelector("#copy-status");
  const label = document.querySelector("#copy-citation span");
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard not available");
    await navigator.clipboard.writeText(text);
    label.textContent = "Copied!";
    status.textContent = "Citation copied to clipboard.";
    setTimeout(() => { label.textContent = "Copy citation"; }, 2500);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Press Ctrl+C or ⌘C to copy.";
  }
});
