"""Trusted UI markup, kept out of Markdown's indented-code parsing."""
import re
from html import escape

def tagline_html():
    examples = (
        "design an aso",
        "evaluate an aso",
        "design a splice-switching aso",
    )
    phrases = "".join(
        f'<span class="aso-design-phrase" style="--letters:{len(text)};'
        f'--delay:{index * 6}s">{escape(text)}</span>'
        for index, text in enumerate(examples)
    )
    return """<style>
.aso-design-examples {display:grid;place-items:center;min-height:2rem;
margin:.25rem auto .65rem;padding:0 .25rem;color:#7CCCDB;
font:600 clamp(.75rem,2.5vw,1.05rem)/1.8 "Courier New",Courier,monospace}
.aso-design-phrase {grid-area:1/1;display:block;box-sizing:content-box;
width:0;visibility:hidden;max-width:100%;overflow:hidden;white-space:nowrap;
border-right:2px solid currentColor;
animation:aso-design-type 18s steps(var(--letters),end) var(--delay) infinite,
aso-design-cursor .75s step-end infinite;animation-play-state:paused}
html.aso-page-ready .aso-design-phrase {animation-play-state:running}
@keyframes aso-design-type {
0% {visibility:visible;width:0}
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
opacity:.35;animation:aso-pulse 2.4s infinite ease-in-out}
.aso-typing-dots span:nth-child(2) {animation-delay:.32s}
.aso-typing-dots span:nth-child(3) {animation-delay:.64s}
.aso-typing-label {font-size:.9rem;opacity:.75}
@keyframes aso-pulse {0%,60%,100% {opacity:.35;transform:translateY(0)}
30% {opacity:1;transform:translateY(-4px)}}
@media (prefers-reduced-motion:reduce) {.aso-typing-dots span {animation:none;opacity:.7}}
</style>
<div class="aso-typing" role="status" aria-live="polite">
<span class="aso-typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>
<span class="aso-typing-label">Working on your request…</span>
</div>"""
