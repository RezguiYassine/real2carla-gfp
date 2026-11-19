// Path and speed plots, drawn when a reader opens "Paths and speeds".
// Plotly (basic bundle, about 1 MB) loads only then.
"use strict";

const PLOTLY_URL = "https://cdn.plot.ly/plotly-basic-2.35.2.min.js";
let plotlyReady = null;

function loadPlotly() {
  plotlyReady ??= new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = PLOTLY_URL;
    s.onload = () => resolve(window.Plotly);
    s.onerror = () => reject(new Error("Plotly did not load"));
    document.head.appendChild(s);
  });
  return plotlyReady;
}

function atTime(series, t) {
  let best = 0;
  series.t.forEach((v, i) => { if (Math.abs(v - t) < Math.abs(series.t[best] - t)) best = i; });
  return best;
}

function draw(Plotly, box, data) {
  const users = data.road_users;
  const speeds = users.flatMap((u) => u.measured.speed);
  const vmax = Math.max(1, ...speeds);
  const top = [], speed = [];
  users.forEach((u, k) => {
    const name = `${k + 1}: ${u.class} ${u.id}`;
    top.push({ x: u.scene.x, y: u.scene.y, mode: "lines", name, line: { color: u.colour, width: 2 },
               hovertemplate: "t = %{text} s<extra>" + name + "</extra>", text: u.scene.t });
    top.push({ x: u.measured.x, y: u.measured.y, mode: "markers", showlegend: false, name: `${name} measured`,
               marker: { size: 6, color: u.measured.speed, cmin: 0, cmax: vmax, colorscale: "Viridis",
                         line: { color: u.colour, width: 1 },
                         colorbar: k === 0 ? { title: { text: "m/s" }, thickness: 10, len: 0.8 } : undefined,
                         showscale: k === 0 },
               text: u.measured.t, hovertemplate: "t = %{text} s, %{marker.color:.1f} m/s<extra></extra>" });
    const i0 = atTime(u.scene, 0);
    top.push({ x: [u.scene.x[i0]], y: [u.scene.y[i0]], mode: "markers", showlegend: false,
               marker: { symbol: "x", size: 11, color: "#222" }, hoverinfo: "skip" });
    speed.push({ x: u.scene.t, y: u.scene.speed, mode: "lines", name, line: { color: u.colour, width: 2 } });
    speed.push({ x: u.measured.t, y: u.measured.speed, mode: "markers", showlegend: false,
                 marker: { size: 5, color: u.colour } });
  });
  const font = { family: "Inter, sans-serif", size: 12 };
  const cfg = { displayModeBar: false, responsive: true };
  Plotly.newPlot(box.querySelector(".plot-top"), top, {
    title: { text: "Top view (× = position at contact)", font: { size: 13 } }, font, margin: { l: 45, r: 10, t: 36, b: 40 },
    xaxis: { title: "x (m)", constrain: "domain" },
    yaxis: { title: "y (m)", scaleanchor: "x", autorange: "reversed", constrain: "domain" },
    legend: { orientation: "h", y: -0.2 }, height: 360,
  }, cfg);
  const [w0, w1] = data.window_s;
  Plotly.newPlot(box.querySelector(".plot-speed"), speed, {
    title: { text: "Speed over time", font: { size: 13 } }, font, margin: { l: 45, r: 10, t: 36, b: 40 },
    xaxis: { title: "t (s), 0 = first contact" }, yaxis: { title: "speed (m/s)", rangemode: "tozero" },
    shapes: [
      { type: "rect", xref: "x", yref: "paper", x0: w0, x1: w1, y0: 0, y1: 1, fillcolor: "#888", opacity: 0.08, line: { width: 0 } },
      { type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: { color: "#c0392b", dash: "dot" } },
    ],
    legend: { orientation: "h", y: -0.2 }, height: 360,
  }, cfg);
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("details.plots").forEach((d) =>
    d.addEventListener("toggle", async () => {
      if (!d.open || d.dataset.drawn) return;
      d.dataset.drawn = "1";
      try {
        const [Plotly, data] = await Promise.all([loadPlotly(), fetch(d.dataset.src).then((r) => r.json())]);
        draw(Plotly, d, data);
      } catch (err) {
        d.querySelector(".plot-row").textContent = `The plots could not be drawn: ${err.message}`;
      }
    }));
});
