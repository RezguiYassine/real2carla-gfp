// Two videos that play as one: the real clip (master) and a reconstructed view
// (slave). Both clips cover the same window around first contact, so the same
// currentTime is the same instant in both.
//
// The slave follows the master's clock. A small drift is corrected by nudging
// the slave's playbackRate (seeking every frame makes browsers stutter); a large
// one, after a stall or a view switch, by a seek. The largest drift seen while
// both play is kept in data-max-drift for the sync check.
"use strict";

const FRAME = 1 / 30;
const SEEK_DRIFT = 0.1;         // s; beyond this the slave seeks instead of catching up
const MAX_NUDGE = 0.15;         // the slave runs at most 15 % faster or slower than the master
const GAIN = 6;                 // rate change per second of drift: 20 ms behind -> 12 % faster

class SyncPlayer {
  static all = [];

  constructor(el) {
    this.el = el;
    this.m = el.querySelector("video.master");
    this.s = el.querySelector("video.slave");
    [this.w0, this.w1] = el.dataset.window.split(",").map(Number);
    this.span = this.w1 - this.w0;
    this.rate = 1;
    this.playing = false;
    this.stalled = null;        // which video is buffering while the other waits
    this.maxDrift = 0;
    this.playBtn = el.querySelector(".play");
    this.scrub = el.querySelector(".scrub");
    this.tlabel = el.querySelector(".tlabel");
    this.caption = el.querySelector(".view-caption");
    el.querySelector(".contact-mark").style.left = `${(-this.w0 / this.span) * 100}%`;
    this.bind();
    SyncPlayer.all.push(this);
  }

  bind() {
    this.playBtn.addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    this.scrub.addEventListener("input", () => this.seek((this.scrub.value / 1000) * this.span));
    this.el.querySelectorAll(".rate").forEach((b) =>
      b.addEventListener("click", () => {
        this.rate = Number(b.dataset.rate);
        this.m.playbackRate = this.s.playbackRate = this.rate;
        this.select(".rate", b);
      }));
    this.el.querySelectorAll(".view-btn").forEach((b) => b.addEventListener("click", () => this.switchView(b)));
    this.m.addEventListener("ended", () => { if (this.playing) this.seek(0, true); });
    for (const [v, other] of [[this.m, this.s], [this.s, this.m]]) {
      v.addEventListener("waiting", () => {
        if (this.playing && this.stalled === null) { this.stalled = v; other.pause(); }
      });
      v.addEventListener("playing", () => {
        // Resume the waiting video without seeking: a seek here would make the
        // other one buffer again, and the two would take turns forever. The
        // rate correction in tick() closes the small gap a stall leaves.
        if (this.stalled === v) { this.stalled = null; if (this.playing) other.play().catch(() => {}); }
      });
    }
    this.m.addEventListener("timeupdate", () => this.show());
  }

  select(selector, button) {
    this.el.querySelectorAll(selector).forEach((b) => b.classList.toggle("is-selected", b === button));
  }

  play() {
    SyncPlayer.all.forEach((p) => { if (p !== this) p.pause(); });
    this.playing = true;
    this.maxDrift = 0;
    this.playBtn.textContent = "❚❚";
    this.playBtn.setAttribute("aria-label", "Pause");
    this.m.playbackRate = this.s.playbackRate = this.rate;
    this.s.currentTime = this.m.currentTime;
    Promise.all([this.m.play(), this.s.play()]).catch(() => {});
    requestAnimationFrame(() => this.tick());
  }

  pause() {
    this.playing = false;
    this.m.pause();
    this.s.pause();
    this.playBtn.textContent = "▶";
    this.playBtn.setAttribute("aria-label", "Play");
    if (this.s.readyState > 0) this.s.currentTime = this.m.currentTime;
  }

  seek(t, keepPlaying = this.playing) {
    this.m.currentTime = t;
    this.s.currentTime = t;
    this.show(t);
    if (keepPlaying) Promise.all([this.m.play(), this.s.play()]).catch(() => {});
  }

  switchView(button) {
    const t = this.m.currentTime;
    this.select(".view-btn", button);
    this.s.poster = button.dataset.poster;
    this.s.src = button.dataset.src;
    if (this.caption) {
      this.caption.textContent = button.dataset.label.startsWith("iteration")
        ? button.dataset.label : `reconstruction: ${button.dataset.label}`;
    }
    this.s.load();
    this.s.addEventListener("loadedmetadata", () => {
      // the master kept playing while the new clip loaded: join it where it is now
      this.s.currentTime = this.playing ? this.m.currentTime : t;
      this.s.playbackRate = this.rate;
      if (this.playing) this.s.play().catch(() => {});
    }, { once: true });
  }

  tick() {
    if (!this.playing) return;
    if (this.stalled === null && !this.m.paused && !this.s.paused && !this.s.seeking) {
      const drift = this.s.currentTime - this.m.currentTime;
      if (Math.abs(drift) > SEEK_DRIFT) {
        this.s.currentTime = this.m.currentTime;
      } else {
        const nudge = Math.max(-MAX_NUDGE, Math.min(MAX_NUDGE, -drift * GAIN));
        const want = this.rate * (1 + nudge);
        if (Math.abs(this.s.playbackRate - want) > 0.005 * this.rate) this.s.playbackRate = want;
        if (this.m.currentTime > 0.5) this.maxDrift = Math.max(this.maxDrift, Math.abs(drift));
        this.el.dataset.maxDrift = this.maxDrift.toFixed(4);
      }
    }
    this.show();
    requestAnimationFrame(() => this.tick());
  }

  show(t = this.m.currentTime) {
    this.scrub.value = Math.round((t / this.span) * 1000);
    const rel = t + this.w0;
    this.tlabel.textContent = `t = ${rel < -0.05 ? "−" : rel > 0.05 ? "+" : ""}${Math.abs(rel).toFixed(1)} s`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".player").forEach((el) => new SyncPlayer(el));
  // Nothing plays off screen: a player scrolled out of view pauses.
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) SyncPlayer.all.find((p) => p.el === e.target)?.pause();
    }
  }, { threshold: 0 });
  SyncPlayer.all.forEach((p) => io.observe(p.el));
});
