"""Trusted UI markup, kept out of Markdown's indented-code parsing."""
import re
from html import escape

SCROLL_EFFECTS = """<script>
(() => {
  const host = window.parent;
  const doc = host.document;
  if (host.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  host.__asoMotionCleanup?.();
  let frame = 0;
  const root = () => doc.querySelector('[data-testid="stMainBlockContainer"]')
    || doc.querySelector('[data-testid="stMain"]');
  const targets = () => {
    const main = root();
    if (!main) return [];
    return [...main.querySelectorAll('.aso-landing-intro > h1, .aso-landing-intro > p, .aso-landing-intro > picture, .aso-scroll-cue, [data-testid="stElementContainer"], .element-container')]
      .filter(el => !el.querySelector('.aso-landing-intro, .aso-startup, style, iframe')
        && !el.closest('[data-testid="stStatusWidget"], [data-testid="stExpander"], [data-testid="stSpinner"]')
        && el.offsetHeight > 0 && !el.closest('[data-testid="stElementContainer"]')?.querySelector('iframe'));
  };
  function update() {
    frame = 0;
    const height = host.innerHeight;
    for (const el of targets()) {
      const rect = el.getBoundingClientRect();
      // Individual translate keeps typing and cursor animations independent.
      const offset = Math.max(-16, Math.min(32, (rect.top / height - .35) * 56));
      el.style.translate = `0 ${offset.toFixed(2)}px`;
    }
  }
  const schedule = () => { if (!frame) frame = host.requestAnimationFrame(update); };
  doc.addEventListener('scroll', schedule, {passive:true, capture:true});
  host.addEventListener('resize', schedule);
  const observer = new host.MutationObserver(schedule);
  if (root()) observer.observe(root(), {childList:true, subtree:true});
  schedule();
  host.__asoMotionCleanup = () => {
    doc.removeEventListener('scroll', schedule, true);
    host.removeEventListener('resize', schedule);
    observer.disconnect();
    if (frame) host.cancelAnimationFrame(frame);
  };
})();
</script>"""

STARTUP_HTML = """<style>
/* Installed before the page content: hide it from its very first paint,
   rather than hiding already-visible content from the later iframe script. */
@media (prefers-reduced-motion:no-preference) {
[data-testid="stMainBlockContainer"]:has(.aso-startup) .aso-landing-intro,
[data-testid="stMainBlockContainer"]:has(.aso-startup)
:is([data-testid="stElementContainer"], .element-container):not(:has(.aso-startup, .aso-landing-intro, style, iframe)) {
animation:aso-content-enter 1.4s ease-out 2s both;
}}
@keyframes aso-content-enter {from {opacity:0;} to {opacity:1;}}
.aso-startup {position:fixed;inset:0;z-index:999990;display:grid;place-items:center;
background:#EAF4F8;color:#253D4B;pointer-events:none;
animation:aso-startup-exit 1.1s ease-in-out 2s forwards;}
.aso-startup-card {width:min(88vw,430px);text-align:center;font-family:Inter,sans-serif;}
.aso-startup-title {font-size:1.6rem;font-weight:900;letter-spacing:-.04em;}
.aso-rna-scene {position:relative;height:150px;margin:16px 0;overflow:hidden;}
.aso-rna-strand {position:absolute;top:88px;left:6%;right:6%;height:5px;
border-radius:8px;background:#0FA3BE;display:flex;justify-content:space-around;}
.aso-rna-strand span {position:relative;top:-11px;font:700 14px/26px 'Courier New',monospace;
background:#EAF4F8;color:#087D94;padding:0 3px;}
.aso-ribosome {position:absolute;left:6%;top:44px;width:86px;height:82px;
animation:aso-ribosome-read 2s ease-in-out forwards;}
.aso-ribosome::before {content:'';position:absolute;left:0;top:0;width:86px;height:46px;
border-radius:48% 55% 24% 30%;background:#164A79;border:2px solid #103B62;
box-shadow:inset 0 5px 0 #ffffff24;}
.aso-ribosome::after {content:'';position:absolute;left:10px;top:52px;width:66px;height:27px;
border-radius:24% 25% 55% 48%;background:#87CDDE;border:2px solid #3295AB;}
.aso-peptide {position:absolute;left:40px;top:-28px;color:#0FA3BE;
font:bold 20px/12px sans-serif;letter-spacing:2px;transform:rotate(-20deg);}
.aso-startup-caption {font-size:.9rem;opacity:.75;}
@keyframes aso-ribosome-read {from {left:6%;} to {left:74%;}}
@keyframes aso-startup-exit {to {opacity:0;visibility:hidden;}}
@media (prefers-reduced-motion:reduce) {
.aso-ribosome {animation:none;left:40%;}
.aso-startup {animation:aso-startup-exit .01s linear .3s forwards;}}
</style>
<div class="aso-startup" role="status" aria-label="Loading MODAL-ASO">
<div class="aso-startup-card"><div class="aso-startup-title">MODAL-ASO</div>
<div class="aso-rna-scene" aria-hidden="true">
<div class="aso-rna-strand"><span>A</span><span>U</span><span>G</span><span>G</span><span>C</span><span>U</span><span>A</span><span>C</span><span>G</span><span>U</span><span>A</span><span>G</span></div>
<div class="aso-ribosome"><span class="aso-peptide">•••</span></div></div>
<div class="aso-startup-caption">Loading your workspace…</div></div></div>"""


def tagline_html():
    examples = (
        "design an aso",
        "evaluate an aso",
        "design a splice-switching aso",
    )
    phrases = "".join(
        f'<span class="aso-design-phrase" style="--letters:{len(text)};'
        f'--delay:{3.3 + index * 6}s">{escape(text)}</span>'
        for index, text in enumerate(examples)
    )
    return """<style>
.aso-design-examples {display:grid;place-items:center;min-height:2rem;
margin:.25rem auto .65rem;padding:0 .25rem;color:#0FA3BE;
font:600 clamp(.75rem,2.5vw,1.05rem)/1.8 "Courier New",Courier,monospace}
.aso-design-phrase {grid-area:1/1;display:block;box-sizing:content-box;
width:0;visibility:hidden;max-width:100%;overflow:hidden;white-space:nowrap;
border-right:2px solid currentColor;
animation:aso-design-type 18s steps(var(--letters),end) var(--delay) infinite,
aso-design-cursor .75s step-end infinite}
@keyframes aso-design-type {
0% {visibility:hidden;width:0}
0.01% {visibility:visible;width:0}
10%,24% {visibility:visible;width:calc(var(--letters) * 1ch)}
32% {visibility:hidden;width:0}
33.33%,100% {visibility:hidden;width:0}}
@keyframes aso-design-cursor {50% {border-color:transparent}}
.aso-design-accessible {position:absolute;width:1px;height:1px;padding:0;
margin:-1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
@media (prefers-reduced-motion:reduce) {
.aso-design-phrase {animation:none;border:0;width:auto;visibility:visible;display:none}
.aso-design-phrase:first-child {display:block}}
</style>
<div class="aso-design-examples">
<span class="aso-design-accessible">Example prompts: design an aso; evaluate an aso; design a splice-switching aso.</span>
<div aria-hidden="true" style="display:contents">""" + phrases + "</div></div>"


def literal_block(value):
    """Display diagnostic syntax literally, including JSON paths and backticks."""
    text = str(value)
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def result_message(result):
    status = str(result.get("status", "finished"))
    if status in {"fatal", "failed", "error"}:
        error = result.get("error") or result.get("last_error") or {}
        if not isinstance(error, dict):
            error = {}
        stage = result.get("failed_stage") or error.get("stage") or "analysis"
        detail = error.get("message") or "The run did not complete. Review its saved diagnostics."
        explanation = ""
        if str(detail).startswith("invalid_structured_output"):
            explanation = "The model's response did not match the required data format.\n\n"
        return f"Analysis stopped during {stage}.\n\n{explanation}{literal_block(detail)}\n\nThe run did not complete; inspect the saved run files for details before retrying."
    if status in {"run_complete", "completed"}:
        return "Analysis completed. Open Results to inspect the report or Files to save the outputs."
    return f"Analysis status: {status}."


TYPING_HTML = """<style>
.aso-typing {display:flex;align-items:center;gap:12px;min-height:32px}
.aso-typing-dots {display:inline-flex;gap:5px;align-items:center}
.aso-typing-dots span {width:7px;height:7px;border-radius:50%;background:currentColor;
opacity:.35;animation:aso-pulse 1.2s infinite ease-in-out}
.aso-typing-dots span:nth-child(2) {animation-delay:.16s}
.aso-typing-dots span:nth-child(3) {animation-delay:.32s}
.aso-typing-label {font-size:.9rem;opacity:.75}
@keyframes aso-pulse {0%,60%,100% {opacity:.35;transform:translateY(0)}
30% {opacity:1;transform:translateY(-4px)}}
@media (prefers-reduced-motion:reduce) {.aso-typing-dots span {animation:none;opacity:.7}}
</style>
<div class="aso-typing" role="status" aria-live="polite">
<span class="aso-typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>
<span class="aso-typing-label">Working on your request…</span>
</div>"""
