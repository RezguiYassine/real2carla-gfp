// Videos that play as one: the real clip (master) and one or more reconstructed
// views (slaves). All clips of an accident cover the same window around first
// contact, so the same currentTime is the same instant in every clip.
//
// Each slave follows the master's clock. A small drift is corrected by nudging
// the slave's playbackRate (seeking every frame makes browsers stutter); a large
// one, after a stall or a view switch, by a seek. While one video buffers, the
// others wait for it. The largest drift seen while all play is kept in
// data-max-drift for the sync check.
//
// A player marked data-autoplay starts by itself whenever it is on screen,
// unless the reader paused it.
"use strict";

const SEEK_DRIFT = 0.1;         // s; beyond this a slave seeks instead of catching up
const MAX_NUDGE = 0.15;         // a slave runs at most 15 % faster or slower than the master
const GAIN = 6;                 // rate change per second of drift: 20 ms behind -> 12 % faster

class SyncPlayer {
  static all = [];

  constructor(el) {
    this.el = el;
    this.m = el.querySelector("video.master");
    this.slaves = [...el.querySelectorAll("video.slave")];
    this.videos = [this.m, ...this.slaves];
    [this.w0, this.w1] = el.dataset.window.split(",").map(Number);
    this.span = this.w1 - this.w0;
    this.rate = 1;
    this.playing = false;
    this.autoplay = "autoplay" in el.dataset;
    this.userPaused = false;
    this.waiting = new Set();   // videos buffering while the others wait
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
    this.playBtn.addEventListener("click", () => {
      this.userPaused = this.playing;
      this.playing ? this.pause() : this.play();
    });
    this.scrub.addEventListener("input", () => this.seek((this.scrub.value / 1000) * this.span));
    this.el.querySelectorAll(".rate").forEach((b) =>
      b.addEventListener("click", () => {
        this.rate = Number(b.dataset.rate);
        this.videos.forEach((v) => { v.playbackRate = this.rate; });
        this.select(".rate", b);
      }));
    this.el.querySelectorAll(".view-btn").forEach((b) => b.addEventListener("click", () => this.switchView(b)));
    this.m.addEventListener("ended", () => { if (this.playing) this.seek(0, true); });
    for (const v of this.videos) {
      v.addEventListener("waiting", () => {
        if (!this.playing) return;
        this.waiting.add(v);
        this.videos.forEach((o) => { if (!this.waiting.has(o)) o.pause(); });
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
    this.waiting.clear();
    this.playBtn.textContent = "❚❚";
    this.playBtn.setAttribute("aria-label", "Pause");
    this.videos.forEach((v) => { v.playbackRate = this.rate; });
    this.slaves.forEach((s) => { s.currentTime = this.m.currentTime; });
    Promise.all(this.videos.map((v) => v.play())).catch((err) => {
      // the browser refused to start without a click: show the play button again
      if (err && err.name === "NotAllowedError") this.pause();
    });
    requestAnimationFrame(() => this.tick());
  }

  pause() {
    this.playing = false;
    this.waiting.clear();
    this.videos.forEach((v) => v.pause());
    this.playBtn.textContent = "▶";
    this.playBtn.setAttribute("aria-label", "Play");
    this.slaves.forEach((s) => { if (s.readyState > 0) s.currentTime = this.m.currentTime; });
  }

  seek(t, keepPlaying = this.playing) {
    this.videos.forEach((v) => { v.currentTime = t; });
    this.show(t);
    if (keepPlaying) Promise.all(this.videos.map((v) => v.play())).catch(() => {});
  }

  switchView(button) {
    const s = this.slaves[0];
    const t = this.m.currentTime;
    this.select(".view-btn", button);
    s.poster = button.dataset.poster;
    s.src = button.dataset.src;
    if (this.caption) {
      this.caption.textContent = button.dataset.label.startsWith("iteration")
        ? button.dataset.label : `reconstruction: ${button.dataset.label}`;
    }
    s.load();
    s.addEventListener("loadedmetadata", () => {
      // the master kept playing while the new clip loaded: join it where it is now
      s.currentTime = this.playing ? this.m.currentTime : t;
      s.playbackRate = this.rate;
      if (this.playing) s.play().catch(() => {});
    }, { once: true });
  }

  tick() {
    if (!this.playing) return;
    this.resume();
    if (this.waiting.size === 0 && !this.m.paused) {
      for (const s of this.slaves) {
        if (s.paused || s.seeking) continue;
        const drift = s.currentTime - this.m.currentTime;
        if (Math.abs(drift) > SEEK_DRIFT) {
          s.currentTime = this.m.currentTime;
          continue;
        }
        const nudge = Math.max(-MAX_NUDGE, Math.min(MAX_NUDGE, -drift * GAIN));
        const want = this.rate * (1 + nudge);
        if (Math.abs(s.playbackRate - want) > 0.005 * this.rate) s.playbackRate = want;
        if (this.m.currentTime > 0.5) this.maxDrift = Math.max(this.maxDrift, Math.abs(drift));
      }
      this.el.dataset.maxDrift = this.maxDrift.toFixed(4);
    }
    this.show();
    requestAnimationFrame(() => this.tick());
  }

  // Once every buffering video has data ahead of it, start whatever was paused
  // for it. Done here rather than on the "playing" event: two videos can report
  // "waiting" at once, and the one paused by the other never fires "playing".
  // No seek on resume: it would make another video buffer again, and they would
  // take turns forever; the rate correction closes the small gap a stall leaves.
  resume() {
    for (const v of this.waiting) {
      if (v.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) this.waiting.delete(v);
    }
    if (this.waiting.size) return;
    for (const v of this.videos) {
      if (v.paused && !v.ended) v.play().catch(() => {});
    }
  }

  show(t = this.m.currentTime) {
    this.scrub.value = Math.round((t / this.span) * 1000);
    const rel = t + this.w0;
    this.tlabel.textContent = `t = ${rel < -0.05 ? "−" : rel > 0.05 ? "+" : ""}${Math.abs(rel).toFixed(1)} s`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".player").forEach((el) => new SyncPlayer(el));
  // Nothing plays off screen: a player scrolled out of view pauses, and an
  // autoplay player scrolled back into view starts again.
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const p = SyncPlayer.all.find((q) => q.el === e.target);
      if (!p) continue;
      if (!e.isIntersecting) p.pause();
      else if (p.autoplay && !p.userPaused && !p.playing) p.play();
    }
  }, { threshold: 0 });
  SyncPlayer.all.forEach((p) => io.observe(p.el));
});
