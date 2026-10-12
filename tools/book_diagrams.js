// Runs inside headless Chrome, on the page written by tools/build_book.py.
//
// Input : window.BOOK_DIAGRAMS = [{ index: 0, code: "flowchart LR ..." }, ...]
//         and the global `mermaid` object from the pinned mermaid.min.js.
// Output: one <pre> element with the id "book-diagram-report" that holds an
//         ASCII-only JSON report.
//         build_book.py reads it from the dumped DOM. If the element is missing,
//         rendering did not finish and the build fails.

const TEXTBOOK_THEME = {
  startOnLoad: false,
  securityLevel: "strict",
  theme: "base",
  fontFamily: '"Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif',
  themeVariables: {
    fontSize: "14px",
    primaryColor: "#eef3f8",
    primaryTextColor: "#1c2733",
    primaryBorderColor: "#4f6f8f",
    lineColor: "#48586a",
    secondaryColor: "#f6f1e4",
    tertiaryColor: "#ffffff",
    edgeLabelBackground: "#ffffff",
  },
  flowchart: {
    htmlLabels: true,
    useMaxWidth: false,
    nodeSpacing: 26,
    rankSpacing: 26,
    padding: 8,
  },
};

/** Render one diagram and return its report entry. */
async function renderDiagram(diagram, host) {
  const entry = { index: diagram.index, ok: false, error: null, svg: null, width: 0, height: 0 };
  try {
    await mermaid.parse(diagram.code);
    const { svg } = await mermaid.render(`book-diagram-${diagram.index}`, diagram.code);
    host.innerHTML = svg;
    const element = host.querySelector("svg");
    if (!element) {
      throw new Error("mermaid returned no <svg> element");
    }
    const box = element.getBBox();
    const view = element.viewBox.baseVal;
    if (box.width < 1 || box.height < 1 || view.width < 1 || view.height < 1) {
      throw new Error("the rendered diagram is empty");
    }
    // Give the figure a fixed natural size; the book's CSS scales it to the page.
    element.removeAttribute("style");
    element.setAttribute("width", String(view.width));
    element.setAttribute("height", String(view.height));
    entry.width = view.width;
    entry.height = view.height;
    entry.svg = host.innerHTML;
    entry.ok = true;
  } catch (error) {
    entry.error = String((error && error.message) || error);
  }
  host.innerHTML = "";
  return entry;
}

/** JSON with every non-ASCII character escaped, so stdout encoding cannot matter. */
function asciiJson(value) {
  return JSON.stringify(value).replace(
    /[\u007f-￿]/g,
    (char) => "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0"),
  );
}

(async () => {
  const report = { userAgent: navigator.userAgent, fatal: null, diagrams: [] };
  try {
    if (typeof mermaid === "undefined") {
      throw new Error("mermaid.js did not load");
    }
    mermaid.initialize(TEXTBOOK_THEME);
    const host = document.getElementById("host");
    for (const diagram of window.BOOK_DIAGRAMS) {
      report.diagrams.push(await renderDiagram(diagram, host));
    }
  } catch (error) {
    report.fatal = String((error && error.message) || error);
  }
  const output = document.createElement("pre");
  output.id = "book-diagram-report";
  output.textContent = asciiJson(report);
  document.body.append(output);
})();
