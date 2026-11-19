// Gallery filters and the BibTeX copy button.
"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const cards = [...document.querySelectorAll("#gallery .card-acc")];
  const twoWheeler = /motorcycle|bicycle/;
  document.querySelectorAll(".filter").forEach((b) =>
    b.addEventListener("click", () => {
      const f = b.dataset.filter;
      document.querySelectorAll(".filter").forEach((x) => x.classList.toggle("is-selected", x === b));
      for (const c of cards) {
        const show = f === "all" || c.dataset.collision === f
          || (f === "two-wheeler" && twoWheeler.test(c.dataset.users));
        c.hidden = !show;
      }
    }));

  const copy = document.querySelector(".copy-bibtex");
  copy?.addEventListener("click", async () => {
    const text = document.getElementById("bibtex-code").textContent;
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Select the text to copy";
    }
    setTimeout(() => { copy.textContent = "Copy"; }, 2000);
  });
});
