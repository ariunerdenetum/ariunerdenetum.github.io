/* Ariun-Erdene Tumurchuluun — site behaviour.
   No dependencies. Four jobs: theme, agenda, reading progress, table wrapping.

   The agenda is built from the DOM rather than written by hand, so it can never
   drift out of sync with the page. Two shapes are supported:
     - section[data-nav] (+ optional data-kicker)  — home and listing pages
     - h2 inside .post-body                        — markdown blog posts
*/
(function () {
  "use strict";

  var root = document.documentElement;

  /* ------------------------------------------------------------- theme ---- */
  /* Same localStorage key and data-theme contract as the demo page, so a
     choice made on either one carries across. */

  function setTheme(t) {
    if (t) root.setAttribute("data-theme", t);
    else root.removeAttribute("data-theme");
    try {
      if (t) localStorage.setItem("theme", t);
      else localStorage.removeItem("theme");
    } catch (e) {}
  }

  function currentTheme() {
    var a = root.getAttribute("data-theme");
    if (a) return a;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  var tog = document.getElementById("theme-tog");
  if (tog) {
    tog.addEventListener("click", function () {
      setTheme(currentTheme() === "dark" ? "light" : "dark");
    });
  }

  /* ------------------------------------------------------------ agenda ---- */

  var el = function (tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  // Collect the targets the agenda will track, in document order.
  function collectTargets() {
    var out = [];
    var sections = document.querySelectorAll("section[data-nav]");
    if (sections.length) {
      Array.prototype.forEach.call(sections, function (s) {
        if (s.id) out.push({ node: s, label: s.dataset.nav, kicker: s.dataset.kicker || "" });
      });
      return out;
    }
    // Posts supply short agenda labels with a kramdown IAL on the heading:
    //   {: data-kicker="Finding 3" data-nav="Prose scales, code doesn't" }
    // Without one the full heading text is used, which is correct but long.
    var heads = document.querySelectorAll(".post-body > h2[id]");
    Array.prototype.forEach.call(heads, function (h) {
      out.push({
        node: h,
        label: h.dataset.nav || h.textContent.trim(),
        kicker: h.dataset.kicker || ""
      });
    });
    return out;
  }

  var targets = collectTargets();
  var links = [];

  function buildList(inMobile) {
    var ol = el("ol");
    targets.forEach(function (t) {
      var a = el("a");
      a.href = "#" + t.node.id;
      if (t.kicker) a.appendChild(el("span", "kk", t.kicker));
      a.appendChild(el("span", null, t.label));
      var li = el("li");
      li.appendChild(a);
      ol.appendChild(li);
      links.push({ a: a, node: t.node, label: t.label, mobile: inMobile });
    });
    return ol;
  }

  var deskMount = document.getElementById("agenda");
  var mobMount = document.getElementById("agenda-m");

  if (targets.length > 1) {
    if (deskMount) {
      deskMount.appendChild(el("p", "toc-title", "On this page"));
      deskMount.appendChild(buildList(false));
    }
    if (mobMount) {
      var d = el("details", "toc-m");
      var sum = el("summary");
      sum.innerHTML =
        '<svg class="chev" viewBox="0 0 16 16" aria-hidden="true">' +
        '<path d="M6 3l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.7" ' +
        'stroke-linecap="round" stroke-linejoin="round"/></svg>' +
        "<span>On this page</span>";
      var now = el("span", "now");
      sum.appendChild(now);
      d.appendChild(sum);
      d.appendChild(buildList(true));
      mobMount.appendChild(d);
      // Tapping a link closes the sheet so the target is actually visible.
      d.addEventListener("click", function (e) {
        if (e.target.closest("a")) d.removeAttribute("open");
      });
      mobMount._now = now;
    }
  } else if (mobMount) {
    mobMount.remove();
  }

  /* Highlight whichever target owns the top third of the viewport. A plain
     rAF-throttled scroll handler beats IntersectionObserver here: sections are
     far taller than the viewport, so threshold callbacks fire unpredictably. */

  function mark() {
    var line = innerHeight * 0.3;
    var active = null; // nothing highlighted while the title is still on screen
    for (var i = 0; i < targets.length; i++) {
      if (targets[i].node.getBoundingClientRect().top <= line) active = targets[i];
    }
    // At the very bottom the last target is what is being read — skipped when
    // the whole page fits on screen, where "the bottom" carries no information.
    var scrollable = document.body.scrollHeight - innerHeight > 200;
    if (scrollable && innerHeight + scrollY >= document.body.scrollHeight - 4) {
      active = targets[targets.length - 1];
    }
    links.forEach(function (l) {
      if (active && l.node === active.node) l.a.setAttribute("aria-current", "true");
      else l.a.removeAttribute("aria-current");
    });
    if (mobMount && mobMount._now) {
      mobMount._now.textContent = active ? "· " + active.label : "";
    }
    if (bar) {
      var max = document.body.scrollHeight - innerHeight;
      bar.style.width = (max > 0 ? Math.min(1, scrollY / max) * 100 : 0) + "%";
    }
  }

  var bar = document.querySelector(".progress");

  /* Coalesce scroll bursts into one measurement per frame. The timer is a safety
     net, not decoration: gating purely on "is a frame pending" means a single
     dropped rAF callback — which happens in throttled contexts like background
     tabs and some webviews — leaves the flag set and kills the agenda for the
     rest of the page's life. Whichever of the two fires first cancels the other. */
  var frame = 0, timer = 0;

  function run() {
    cancelAnimationFrame(frame);
    clearTimeout(timer);
    frame = timer = 0;
    mark();
  }

  function onScroll() {
    if (frame || timer) return;
    frame = requestAnimationFrame(run);
    timer = setTimeout(run, 150);
  }

  if (targets.length > 1 || bar) {
    addEventListener("scroll", onScroll, { passive: true });
    addEventListener("resize", onScroll);
    mark();
  }

  /* ------------------------------------------- wrap tables so they scroll -- */
  /* Kramdown emits bare <table>; without this the post's wide comparison
     tables would scroll the whole page sideways on a phone. */

  Array.prototype.forEach.call(document.querySelectorAll(".post-body table"), function (t) {
    if (t.parentNode.classList.contains("scroller")) return;
    var w = el("div", "scroller");
    t.parentNode.insertBefore(w, t);
    w.appendChild(t);
  });
})();
