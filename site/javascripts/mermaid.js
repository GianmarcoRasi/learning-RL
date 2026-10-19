// Render ```mermaid fences with Mermaid's own dark theme so the lessons' classDef
// colours (light text on dark nodes) stay readable in both site colour schemes.
const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.12.0/dist/mermaid.min.js";
let mermaidLoaded;

function loadMermaid() {
  mermaidLoaded ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = MERMAID_URL;
    script.onload = () => {
      window.mermaid.initialize({ startOnLoad: false, theme: "dark" });
      resolve(window.mermaid);
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
  return mermaidLoaded;
}

document$.subscribe(async () => {
  const diagrams = document.querySelectorAll(".rl-mermaid");
  if (!diagrams.length) return;
  const mermaid = await loadMermaid();
  await mermaid.run({ nodes: diagrams });
});
