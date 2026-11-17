"""Optional local browser UI. No scientific decisions live here."""
from pathlib import Path
from queue import Empty
from itertools import groupby
from base64 import b64encode
import streamlit as st
import streamlit.components.v1 as components
from aso_agents.gui.config import DEFAULT_OPENROUTER_MODEL, Settings, SetupError, list_models, efforts, validate, environment_for, configuration_preview
from aso_agents.gui.worker import start
from aso_agents.gui.presentation import SCROLL_EFFECTS, STARTUP_HTML, TYPING_HTML, result_message, tagline_html

def branding():
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
:root {{color-scheme:light;}}
[data-testid="stMainBlockContainer"]:has(.aso-landing-intro) {{padding-top:2rem;}}
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] :is(h1,h2,h3,h4,h5,h6,p,label,input,textarea,button,select),
[data-baseweb="select"], [data-baseweb="popover"], [role="listbox"] {{
font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}}
[data-testid="stChatMessage"], [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatInput"], [data-testid="stChatInput"] textarea,
[data-testid="stSidebar"] {{font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}}
.aso-page-title {{font-family:"Inter",sans-serif !important;font-weight:900 !important;
font-size:clamp(2.3rem,5vw,3.6rem) !important;letter-spacing:-0.04em;padding-bottom:0.5rem;}}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"] {{background:#EAF4F8;color:#253D4B;}}
[data-testid="stSidebar"], [data-testid="stBottom"] {{background:#F6FAFC;color:#253D4B;}}
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stWidgetLabel"] p {{color:#253D4B;}}
[data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="select"] > div {{background:#F6FAFC;color:#253D4B;border-color:#C5DDE7;}}
[data-baseweb="select"] > div:focus-within {{border-color:#0FA3BE;box-shadow:0 0 0 1px #0FA3BE;}}
[data-baseweb="popover"], [role="listbox"] {{background:#F6FAFC;color:#253D4B;}}
[role="option"][aria-selected="true"], [role="option"]:hover {{background:#D8F1F6 !important;color:#146579 !important;}}
[data-baseweb="select"] [data-baseweb="tag"] {{background:#0FA3BE;color:white;}}
button[kind="primary"] {{background:#0FA3BE;border-color:#0FA3BE;color:white;}}
input {{accent-color:#0FA3BE;}}
input, textarea {{color:#253D4B !important;background:#F6FAFC !important;}}
.aso-setup-loading {{padding:16px 4px 12px;margin-top:8px;}}
.aso-scroll-cue {{display:flex;flex-direction:row;justify-content:center;align-items:center;gap:8px;
margin:.5rem auto 0;color:#146579 !important;text-decoration:none !important;
font-size:.85rem;font-weight:500;}}
.aso-scroll-cue span {{font-size:1.5rem;line-height:1;}}
#aso-get-started {{scroll-margin-top:5rem;}}
@media (prefers-reduced-motion:no-preference) {{

.aso-scroll-cue span {{animation:aso-scroll-nudge 1.8s ease-in-out infinite;}}
.aso-scroll-cue span:nth-child(2) {{animation-delay:.18s;}}
.aso-scroll-cue span:nth-child(3) {{animation-delay:.36s;}}
html, [data-testid="stMain"] {{scroll-behavior:smooth;}}
}}
@keyframes aso-scroll-nudge {{0%,100% {{transform:translateY(0);}} 50% {{transform:translateY(7px);}}}}
.aso-landing-intro > picture {{display:block;}}
</style>""", unsafe_allow_html=True)

def introduction():
    assets = Path(__file__).resolve().parents[1] / "resources" / "gui"
    desktop = b64encode((assets / "mechanisms.png").read_bytes()).decode("ascii")
    mobile = b64encode((assets / "mechanisms-mobile.png").read_bytes()).decode("ascii")
    st.markdown(f"""
<section class="aso-landing-intro" style="max-width:1100px;margin:0 auto .25rem;text-align:center;">
<h1 class="aso-page-title">MODAL-ASO</h1>
<p style="font-size:1.15rem;font-weight:600;line-height:1.6;margin:0 auto 0.7rem;max-width:800px;">Multi-mechanism Orchestration and Design by an Agentic LLM for ASOs</p>
{tagline_html()}
<picture>
<source media="(max-width:640px)" srcset="data:image/png;base64,{mobile}">
<img src="data:image/png;base64,{desktop}" alt="ASO mechanisms: RNase H-mediated RNA degradation, splice modulation, and steric blocking of regulatory interactions." style="display:block;width:100%;height:auto;max-height:26vh;object-fit:contain;margin:auto;border-radius:12px;">
</picture>
<a class="aso-scroll-cue" href="#aso-get-started" target="_self" aria-label="Scroll to get started"><span aria-hidden="true">↓</span><span aria-hidden="true">↓</span><span aria-hidden="true">↓</span></a>
</section>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="MODAL-ASO", layout="wide")
    branding()
    state = st.session_state
    if not state.get("startup_shown"):
        st.markdown(STARTUP_HTML, unsafe_allow_html=True)
        state.startup_shown = True

    if "validated" in state:
        try:
            environment_for(state.validated)
        except SetupError:
            state.pop("validated")
            st.warning("Reconfigure this session using the existing CLI settings. Earlier chat messages are retained.")

    if "validated" not in state:
        introduction()
        st.markdown('<h2 id="aso-get-started">Connect your language model</h2>', unsafe_allow_html=True)
        components.html(SCROLL_EFFECTS, height=0)
        provider = st.selectbox("LLM provider", ["openrouter", "openai"])
        key = st.text_input("API key", type="password", key=f"key_{provider}")
        identity = (provider, key.strip())
        if state.get("catalog_identity") != identity:
            state.pop("models", None)
        if provider == "openrouter":
            refresh_models = st.button("Load compatible models", disabled=not key.strip())
            if refresh_models or (key.strip() and state.get("catalog_attempt_identity") != identity):
                state.catalog_attempt_identity = identity
                loading_models = st.empty()
                loading_models.markdown(
                    TYPING_HTML.replace("Working on your request…", "Loading model…")
                    .replace('class="aso-typing"', 'class="aso-typing aso-setup-loading"'),
                    unsafe_allow_html=True,
                )
                try:
                    state.models = list_models(provider, key)
                    state.catalog_identity = identity
                except SetupError as exc:
                    st.error(f"{exc.field}: {exc}")
                    loading_models.empty()
                else:
                    loading_models.empty()
        else:
            st.info("OpenAI uses the existing per-agent CLI model and reasoning settings. Global GUI overrides are unavailable.")

        if key.strip():
            models = state.get("models", {})
            if provider == "openrouter":
                model_options = list(dict.fromkeys([DEFAULT_OPENROUTER_MODEL, *models]))
                model = st.selectbox("Model", model_options,
                    key="selected_router_model")
                options = models[model] if isinstance(models, dict) and model in models else efforts(model)
                options = [value for value in options if value != "default"] or ["default"]
                effort = st.selectbox("Reasoning effort", options,
                    index=len(options) - 1,
                    format_func=lambda value: "Provider default (no supported effort levels advertised)" if value == "default" else value,
                    key=f"selected_router_effort_{model}_{'_'.join(options)}")
                timeout, retries = None, None
                with st.expander("Advanced · timeout and retries"):
                    st.caption("Leave controls unchecked to inherit existing CLI settings. Network timeouts are per API call, not a total analysis deadline.")
                    if st.checkbox("Set network timeout"):
                        timeout = st.number_input("Network timeout (seconds)", min_value=1.0, value=120.0, step=10.0)
                    if st.checkbox("Set SDK retry count"):
                        retries = st.number_input("SDK retries", min_value=0, value=0, step=1)
                    st.caption("Any additional framework transport retries retain their existing CLI configuration.")
                settings = Settings(provider, key.strip(), model, effort, timeout, retries)
            else:
                settings = Settings(provider, key.strip())

            configuration_ok = False
            try:
                configuration_preview(settings)
                configuration_ok = True
            except (SetupError, ValueError, RuntimeError) as exc:
                st.error(f"Configuration: {exc}")
            st.caption("OpenRouter starts with DeepSeek Flash. Reasoning effort starts at the highest supported level for the selected model; you can change it.")
            with st.expander("What validation checks and how long it takes"):
                st.markdown("""
1. **Settings:** validate the existing CLI provider configuration and any explicit model/effort selections.
2. **Authentication and model access:** send one small probe through the existing provider transport.
3. **Tools and structured output:** request a structured test-tool call.
4. **Reasoning settings:** the provider applies explicit or inherited overrides using the same configuration code as the CLI.

**Expected wait:** roughly 5–30 seconds; this is an estimate. The setup probe uses a 40-second network timeout and no automatic retries. Its timeout does not change analysis settings.

This checks one small configuration request. Per-agent CLI defaults remain in effect; validation does not exercise every agent, run an ASO analysis, or guarantee that every later request will succeed.""")
            if st.button("Start Chat", type="primary", disabled=not configuration_ok):
                validation_error = None
                loading_model = st.empty()
                loading_model.markdown(
                    TYPING_HTML.replace("Working on your request…", "Checking API key…")
                    .replace('class="aso-typing"', 'class="aso-typing aso-setup-loading"'),
                    unsafe_allow_html=True,
                )
                try:
                    validate(settings, models)
                except SetupError as exc:
                    validation_error = exc
                if validation_error is not None:
                    st.error(f"{validation_error.field}: {validation_error}")
                    loading_model.empty()
                if validation_error is None:
                    state.validated = settings
                    state.setdefault("messages", [])
                    st.rerun()
        st.stop()

    st.markdown('<h1 class="aso-page-title">MODAL-ASO</h1>', unsafe_allow_html=True)
    preview = configuration_preview(state.validated)
    st.caption(f"{preview['Provider']} · {preview['Model']} · reasoning: {preview['Reasoning effort']}")
    if st.sidebar.button("Change provider settings", disabled="job" in state):
        state.pop("validated")
        st.rerun()
    st.sidebar.checkbox("Show technical activity", value=False, key="show_activity")


    @st.fragment(run_every="1s" if "job" in state else None)
    def workspace():
        if "job" in state:
            process, events, answers = state.job
            while True:
                try:
                    kind, value = events.get_nowait()
                except Empty:
                    break
                if kind == "questions":
                    state.questions = value["questions"]
                    context = value.get("context") or {}
                    explanation = "\n\n".join(str(context[field]) for field in ("why_this_matters", "recommendation") if context.get(field))
                    text = "I need your input to continue:\n\n" + "\n".join(f"- {question}" for question in value["questions"])
                    state.messages.append(("assistant", text + ("\n\n" + explanation if explanation else "")))
                elif kind == "activity":
                    state.messages.append(("activity", value))
                elif kind == "message":
                    state.messages.append(("assistant", value))
                elif kind == "result":
                    state.result = value
                    state.messages.append(("assistant", result_message(value)))
                elif kind == "error":
                    state.messages.append(("assistant", value))
                elif kind == "diagnostic":
                    state.messages.append(("diagnostic", value))
                elif kind == "done":
                    state.pop("job", None)
                    state.pop("questions", None)
                    st.rerun()
            if "job" in state and not process.is_alive() and process.exitcode not in (None, 0):
                state.messages.append(("assistant", "The analysis worker stopped unexpectedly. Check the local terminal."))
                state.pop("job")
                st.rerun()
        chat, results, files = st.tabs(["Chat", "Results", "Files"])
        with chat:
            for is_activity, group in groupby(state.messages, key=lambda item: item[0] == "activity"):
                if is_activity:
                    if not state.get("show_activity", False):
                        continue
                    updates = list(group)
                    with st.chat_message("assistant"):
                        with st.expander(f"Activity · {len(updates)} updates", expanded=False):
                            for _, message in updates:
                                st.caption(message)
                else:
                    for role, message in group:
                        with st.chat_message("assistant" if role == "diagnostic" else role):
                            if role == "diagnostic":
                                with st.expander(f"Technical details · {message['type']}"):
                                    st.code(message["traceback"], language="text")
                            else:
                                st.write(message)
            if "questions" in state:
                with st.form("clarifications"):
                    values = [st.text_input(question, key=f"answer_{index}_{question}") for index, question in enumerate(state.questions)]
                    if st.form_submit_button("Send answers"):
                        if all(value.strip() for value in values):
                            state.job[2].put([{"question": q, "answer": a} for q, a in zip(state.questions, values)])
                            state.messages.append(("user", "\n".join(values)))
                            state.pop("questions")
                            st.rerun()
                        else:
                            st.error("Answer each question before continuing.")
            if "job" in state:
                if "questions" not in state:
                    with st.chat_message("assistant"):
                        st.markdown(TYPING_HTML, unsafe_allow_html=True)
                else:
                    st.caption("Waiting for your answers.")
            request = st.chat_input("Describe your ASO design or evaluation request", disabled="job" in state)
            if request:
                state.messages.append(("user", request))
                state.pop("result", None)
                state.job = start(state.validated, request)
                st.rerun()
        root_value = state.get("result", {}).get("run_dir")
        root = Path(root_value).resolve() if root_value else None
        with results:
            report = root / "report.html" if root else None
            if report and report.is_file():
                st.components.v1.html(report.read_text(encoding="utf-8"), height=850, scrolling=True)
            else:
                st.info("The HTML report will appear when available.")
        with files:
            if root and root.is_dir():
                # List run outputs; deferred downloads avoid preloading every file.
                paths = sorted(p for p in root.rglob("*") if p.is_file() and p.resolve().is_relative_to(root)
                               and not any(part.startswith(".") for part in p.relative_to(root).parts))
                for path in paths:
                    relative = str(path.relative_to(root))
                    label, download = st.columns([4, 1])
                    with label:
                        st.write(relative)
                        st.caption(f"{path.stat().st_size:,} bytes")
                    with download:
                        st.download_button("Download", path.read_bytes, file_name=path.name,
                                           key=f"download_{root}_{relative}")
                    if path.suffix == ".md":
                        with st.expander(f"Preview {relative}"):
                            if path.stat().st_size <= 1024 * 1024:
                                st.markdown(path.read_text(encoding="utf-8"))
                            else:
                                st.caption("Download this file to read the full Markdown; it exceeds the 1 MB preview limit.")
                if not paths:
                    st.info("No downloadable output files are available yet.")
                st.caption("Files are read when you click Download. Keep the complete run folder for relative report links.")
            else:
                st.info("Saved outputs will appear here.")


    workspace()


if __name__ == "__main__":
    main()
