"""Optional local browser UI. No scientific decisions live here."""
from pathlib import Path
from queue import Empty
from itertools import groupby
from base64 import b64encode
import sys
from importlib import reload
import streamlit as st
from aso_agents.gui.config import DEFAULT_OPENROUTER_MODEL, Settings, SetupError, list_models, efforts, validate, environment_for, configuration_preview
from aso_agents.gui.worker import start
from aso_agents.gui.presentation import TYPING_HTML, result_message, tagline_html
from aso_agents.gui import loading

# Streamlit reruns this page but retains imported modules between reruns.
# Refresh the browser handshake together with changes to the page styling.
reload(loading)
STARTUP_HTML, STARTUP_READY = loading.STARTUP_HTML, loading.STARTUP_READY

def branding():
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
:root {{color-scheme:light;--background-color:#F5F8FC;--secondary-background-color:#F5F8FC;}}
html, body, [data-testid="stApp"], .stApp {{background:#F5F8FC !important;}}
[data-testid="stMainBlockContainer"]:has(.aso-landing-intro) {{padding-top:2rem;}}
[data-testid="stMainBlockContainer"]:has(.aso-workspace) {{padding-top:.75rem;padding-bottom:.75rem;}}
.aso-workspace {{font-size:clamp(1.5rem,3vw,2.5rem) !important;margin-bottom:0 !important;}}
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {{overflow-anchor:none;}}
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
[data-testid="stHeader"] {{background:#F5F8FC !important;color:#253D4B;}}
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {{background:#F5F8FC !important;color:#253D4B;}}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{background:#E3EAF3 !important;color:#253D4B;}}
[data-testid="stSidebar"] {{border-right:1px solid #C8D4E3;}}
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stWidgetLabel"] p {{color:#253D4B;}}
[data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="select"] > div {{background:#FFFFFF !important;color:#253D4B;border:1px solid #9CB5C7 !important;border-radius:10px;}}
[data-baseweb="input"], [data-baseweb="base-input"] {{background:#FFFFFF !important;border-color:#9CB5C7 !important;border-radius:10px;}}
[data-testid="stTextInput"] input {{background:#FFFFFF !important;}}
[data-testid="stTextInput"] [data-baseweb="input"] {{overflow:hidden;background:#FFFFFF !important;}}
[data-testid="stTextInput"] [data-baseweb="base-input"] {{border:0 !important;border-radius:0;}}
[data-testid="stTextInput"] button,
[data-testid="stTextInput"] [data-baseweb="input"] > div {{background:#FFFFFF !important;}}
[data-testid="stElementContainer"], .element-container,
.aso-landing-intro > h1, .aso-landing-intro > p,
.aso-landing-intro > picture, .aso-scroll-cue {{translate:none !important;}}
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
[data-testid="stSelectbox"] input {{background-color:#FFFFFF !important;}}
[data-baseweb="select"] > div:focus-within {{border-color:#0FA3BE;box-shadow:0 0 0 1px #0FA3BE;}}
[data-baseweb="popover"], [role="listbox"] {{background:#F5F8FC;color:#253D4B;}}
[role="option"][aria-selected="true"], [role="option"]:hover {{background:#D8F1F6 !important;color:#146579 !important;}}
[data-baseweb="select"] [data-baseweb="tag"] {{background:#0FA3BE;color:white;}}
button[kind="primary"] {{background:#0FA3BE;border-color:#0FA3BE;color:white;}}
input {{accent-color:#0FA3BE;}}
input, textarea {{color:#253D4B !important;background:#F5F8FC !important;}}
[data-testid="stTextInputRootElement"],
[data-testid="stSelectboxRootElement"],
[data-testid="stSelectbox"] [data-baseweb="select"] {{
background:#FFFFFF !important;border:1px solid #9CB5C7 !important;
border-radius:10px !important;overflow:hidden;box-sizing:border-box;width:100%;}}
[data-testid="stTextInputRootElement"] :is(div,input,button),
[data-testid="stSelectboxRootElement"] :is(div,input,button),
[data-testid="stSelectbox"] [data-baseweb="select"] :is(div,input,button) {{
background:#FFFFFF !important;border:0 !important;border-radius:0 !important;box-shadow:none !important;}}
[data-testid="stTextInputRootElement"] button {{align-self:stretch;margin:0 !important;padding:0 12px !important;}}
[data-testid="stTextInput"] [data-testid="InputInstructions"] {{
position:static !important;inset:auto !important;transform:none !important;
display:block !important;width:100%;height:auto;margin-top:6px;
padding:0 2px;background:transparent !important;text-align:right;}}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stSelectboxRootElement"]:focus-within {{outline:2px solid #0FA3BE;outline-offset:1px;}}
.st-key-load_compatible_models button,
.st-key-load_compatible_models button:is(:hover,:focus,:active,:disabled) {{
background:#0FA3BE !important;border-color:#0FA3BE !important;color:#FFFFFF !important;opacity:1 !important;}}
.st-key-load_compatible_models button p {{color:#FFFFFF !important;}}
[data-testid="stSelectbox"] div:has(> input[role="combobox"]) {{
display:flex;align-items:center;width:100%;background:#FFFFFF !important;
border:1px solid #9CB5C7 !important;border-radius:10px;overflow:hidden;}}
[data-testid="stSelectbox"] input[role="combobox"] {{
background:transparent !important;border:0 !important;flex:1;min-width:0;}}
[data-testid="stSelectbox"] button[aria-label="Open"],
[data-testid="stSelectbox"] button[aria-label="Clear value"] {{
background:#FFFFFF !important;border:0 !important;align-self:stretch;margin:0;padding:0 12px;}}
[data-testid="stExpander"] details {{background:#FCFDFF;border:1px solid #DCE5EF;border-radius:10px;}}
[data-testid="stExpander"] summary {{background:#FCFDFF;border-radius:10px;}}
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
html:has(.aso-landing-intro), [data-testid="stMain"]:has(.aso-landing-intro) {{scroll-behavior:smooth;}}
}}
@keyframes aso-scroll-nudge {{0%,100% {{transform:translateY(0);}} 50% {{transform:translateY(7px);}}}}
.aso-landing-intro > picture {{display:block;position:relative;isolation:isolate;padding:44px 60px;margin:6px auto 10px;}}
.aso-landing-intro > picture::after {{content:"";position:absolute;inset:12% 3% 4%;z-index:-2;pointer-events:none;
background:radial-gradient(ellipse at center,rgba(88,117,143,.12) 20%,rgba(88,117,143,.05) 55%,transparent 78%);filter:blur(14px);transform:translateY(5px);}}
.aso-landing-intro > picture::before {{content:"";position:absolute;inset:-8% -5%;z-index:-1;background-image:var(--aso-paint);background-size:100% 100%;background-repeat:no-repeat;
filter:brightness(1.04) blur(5px);
mask-image:radial-gradient(ellipse at center,#000 40%,rgba(0,0,0,.85) 58%,rgba(0,0,0,.3) 78%,transparent 98%);
-webkit-mask-image:radial-gradient(ellipse at center,#000 40%,rgba(0,0,0,.85) 58%,rgba(0,0,0,.3) 78%,transparent 98%);}}
.aso-landing-intro > picture > img {{mix-blend-mode:multiply;border-radius:0 !important;}}
@media (max-width:640px) {{.aso-landing-intro > picture {{padding:36px 28px;}}}}
[data-baseweb="tab-panel"]:has(.aso-chat-surface), [role="tabpanel"]:has(.aso-chat-surface),
[role="tabpanel"]:has(.aso-output-surface) {{background:#FFFFFF;border:1px solid #B8C8DA;border-radius:12px;
padding:12px;margin:8px 0 0;box-shadow:none;box-sizing:border-box;}}
[data-testid="stChatMessage"] {{background:transparent;border:0;}}
[data-testid="stLayoutWrapper"]:has(> .st-key-chat_history),
.st-key-chat_history {{height:var(--aso-chat-height,min(32dvh,260px)) !important;min-height:0 !important;max-height:none !important;flex:none !important;}}
.st-key-chat_history {{overflow-y:auto !important;overscroll-behavior:contain;overflow-anchor:none;}}
[data-testid="stLayoutWrapper"]:has(> .st-key-results_content), .st-key-results_content {{height:var(--aso-results-height,min(50dvh,400px)) !important;min-height:0 !important;flex:none !important;}}
[data-testid="stLayoutWrapper"]:has(> .st-key-files_content), .st-key-files_content {{height:var(--aso-files-height,min(50dvh,400px)) !important;min-height:0 !important;flex:none !important;}}
.st-key-results_content {{overflow:hidden !important;}}
.st-key-files_content {{overflow:auto !important;overscroll-behavior:contain;overflow-anchor:none;}}
.st-key-files_content [data-testid="stExpander"] details {{overflow:hidden;}}
.st-key-files_content [data-testid="stExpander"] details > div {{min-width:0;}}
.st-key-files_content table {{display:block;max-width:100%;overflow-x:auto;}}
.st-key-files_content [data-testid="stMarkdownContainer"] {{min-width:0;overflow-wrap:anywhere;}}
@media (prefers-reduced-motion:no-preference) {{.st-key-chat_history {{scroll-behavior:smooth;}}}}
[data-testid="stChatInput"], [data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] > div {{background:#E8F0F8 !important;color:#253D4B;}}
[data-testid="stChatInput"] {{border:1px solid #AFC3D9;border-radius:12px;}}
@media (max-width:640px) {{[data-baseweb="tab-panel"]:has(.aso-chat-surface), [role="tabpanel"]:has(.aso-chat-surface), [role="tabpanel"]:has(.aso-output-surface) {{padding:10px;margin:8px 0 0;}}}}
[data-baseweb="tab-list"] {{background:#F5F8FC;border-radius:12px;padding:6px;gap:6px;}}
[data-baseweb="tab"] {{color:#253D4B;border-radius:8px;padding:10px 24px;}}
[data-baseweb="tab"][aria-selected="true"] {{background:#146579;color:#FFFFFF;}}
[data-baseweb="tab"][aria-selected="true"] p {{color:#FFFFFF !important;}}
[data-baseweb="tab-highlight"] {{background:#0FA3BE;}}
[data-testid="stAppViewContainer"]:has(.aso-workspace),
[data-testid="stAppViewContainer"]:has(.aso-workspace) [data-testid="stMain"],
body:has(.aso-workspace) [data-testid="stHeader"],
body:has(.aso-workspace) [data-testid="stBottom"] {{background:#F5F8FC;}}
</style>""", unsafe_allow_html=True)

def introduction():
    assets = Path(__file__).resolve().parents[1] / "resources" / "gui"
    desktop = b64encode((assets / "mechanisms.png").read_bytes()).decode("ascii")
    mobile = b64encode((assets / "mechanisms-mobile.png").read_bytes()).decode("ascii")
    paint = b64encode((assets / "paint-backdrop.png").read_bytes()).decode("ascii")
    st.markdown(f"""
<section class="aso-landing-intro" style="max-width:1100px;margin:0 auto .25rem;text-align:center;">
<h1 class="aso-page-title">MODAL-ASO</h1>
<p style="font-size:1.15rem;font-weight:600;line-height:1.6;margin:0 auto 0.7rem;max-width:800px;">Multi-mechanism Orchestration and Design by an Agentic LLM for ASOs</p>
{tagline_html()}
<picture style="--aso-paint:url('data:image/png;base64,{paint}')">
<source media="(max-width:640px)" srcset="data:image/png;base64,{mobile}">
<img src="data:image/png;base64,{desktop}" alt="ASO mechanisms: RNase H-mediated RNA degradation, splice modulation, and steric blocking of regulatory interactions." style="display:block;width:100%;height:auto;max-height:26vh;object-fit:contain;margin:auto;border-radius:12px;">
</picture>
<a class="aso-scroll-cue" href="#aso-get-started" target="_self" aria-label="Scroll to get started"><span aria-hidden="true">↓</span><span aria-hidden="true">↓</span><span aria-hidden="true">↓</span></a>
</section>
""", unsafe_allow_html=True)

def main():
    models_loaded = False
    test_mode = "--test" in sys.argv
    load_models, check_key, preview_settings, start_job = list_models, validate, configuration_preview, start
    if test_mode:
        from aso_agents.gui import mock
        load_models, check_key, preview_settings, start_job = mock.list_models, mock.validate, mock.configuration_preview, mock.start
    st.set_page_config(page_title="MODAL-ASO", layout="wide")
    branding()
    state = st.session_state
    if test_mode:
        st.caption("UI test mode · Offline mock responses · API key: correctkey")
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
        provider = st.selectbox("LLM provider", ["openrouter", "openai"])
        key = st.text_input("API key", type="password", key=f"key_{provider}")
        identity = (provider, key.strip())
        if state.get("catalog_identity") != identity:
            state.pop("models", None)
        if provider == "openrouter":
            refresh_models = st.button("Load compatible models", disabled=not key.strip(), key="load_compatible_models")
            if refresh_models or (key.strip() and state.get("catalog_attempt_identity") != identity):
                state.catalog_attempt_identity = identity
                loading_models = st.empty()
                loading_models.markdown(
                    TYPING_HTML.replace("Working on your request…", "Loading model…")
                    .replace('class="aso-typing"', 'class="aso-typing aso-setup-loading"'),
                    unsafe_allow_html=True,
                )
                try:
                    state.models = load_models(provider, key)
                    state.catalog_identity = identity
                except SetupError as exc:
                    st.error(f"{exc.field}: {exc}")
                    loading_models.empty()
                else:
                    loading_models.empty()
                    models_loaded = True
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
                preview_settings(settings)
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
                    check_key(settings, models)
                except SetupError as exc:
                    validation_error = exc
                if validation_error is not None:
                    st.error(f"{validation_error.field}: {validation_error}")
                    loading_model.empty()
                if validation_error is None:
                    state.validated = settings
                    state.setdefault("messages", [])
                    st.rerun()
        st.iframe(STARTUP_READY, height=1)
        if models_loaded:
            st.iframe("""<script>
const host = window.parent;
host.requestAnimationFrame(() => host.requestAnimationFrame(() => {
  const section = host.document.querySelector('.st-key-selected_router_model');
  if (section) {
    section.style.scrollMarginTop = '5rem';
    section.scrollIntoView({block:'start', behavior:host.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  }
}));
</script>""", height=1)
        st.stop()

    st.markdown('<h1 class="aso-page-title aso-workspace">MODAL-ASO</h1>', unsafe_allow_html=True)
    preview = preview_settings(state.validated)
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
                    if value not in state.get("seen_commentary", set()):
                        state.setdefault("seen_commentary", set()).add(value)
                        state.messages.append(("assistant", value))
                elif kind == "result":
                    state.result = value
                    message = (f"UI preview ready. Results and Files show saved outputs from {value['preview_source']}. No new analysis was run."
                               if test_mode and value.get('preview_source') else result_message(value))
                    state.messages.append(("assistant", message))
                elif kind == "error":
                    state.messages.append(("assistant", value))
                elif kind == "diagnostic":
                    if value.get("type") == "SDK serialization warning":
                        summary = state.get("sdk_warning_summary")
                        if summary is None:
                            summary = {"type": "SDK serialization warning", "severity": "warning", "count": 0,
                                       "samples": [], "traceback": ""}
                            state.sdk_warning_summary = summary
                            state.messages.append(("diagnostic", summary))
                        summary["count"] += 1
                        detail = value.get("traceback") or value.get("message", "")
                        if detail not in summary["samples"] and len(summary["samples"]) < 10:
                            summary["samples"].append(detail)
                        summary["traceback"] = "\n\n".join(summary["samples"])
                    else:
                        state.messages.append(("diagnostic", value))
                elif kind == "done":
                    state.pop("job", None)
                    state.pop("questions", None)
                    st.rerun()
            if "job" in state and not process.is_alive() and process.exitcode not in (None, 0):
                state.messages.append(("assistant", "The analysis worker stopped unexpectedly. Check the local terminal."))
                state.pop("job")
                st.rerun()
        chat, results, files = st.tabs(["Chat", "Results", "Files"], key="workspace_tab", on_change="rerun")
        with chat:
            st.markdown('<span class="aso-chat-surface" aria-hidden="true" style="display:none"></span>', unsafe_allow_html=True)
            history = st.container(height=460, border=False, key="chat_history", autoscroll=True)
            with history:
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
                            if role == "diagnostic" and message.get("type") == "SDK serialization warning" and not state.get("show_activity", False):
                                continue
                            with st.chat_message("assistant" if role == "diagnostic" else role):
                                if role == "diagnostic":
                                    with st.expander(f"Technical details · {message['type']}"):
                                        if message.get("type") == "SDK serialization warning":
                                            st.caption(f"{message.get('count', 1)} warning(s) recorded during this request. The SDK reported unexpected data while serializing.")
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
                if state.get("result") and "job" not in state:
                    def open_output(tab):
                        state.workspace_tab = tab
                    view_results, view_files = st.columns(2)
                    view_results.button("View Results", on_click=open_output, args=("Results",), width="stretch")
                    view_files.button("View Files", on_click=open_output, args=("Files",), width="stretch")
            request = st.chat_input("Describe your ASO design or evaluation request", disabled="job" in state)
            if request:
                state.messages.append(("user", request))
                state.pop("result", None)
                state.seen_commentary = set()
                state.pop("sdk_warning_summary", None)
                with history:
                    with st.chat_message("user"):
                        st.write(request)
                    with st.chat_message("assistant"):
                        st.markdown(TYPING_HTML, unsafe_allow_html=True)
                state.job = start_job(state.validated, request)
                if test_mode:
                    state.mock_output_directory = state.job[0].output_directory
                st.rerun()
        root_value = state.get("result", {}).get("run_dir")
        root = Path(root_value).resolve() if root_value else None
        with results:
            st.markdown('<span class="aso-output-surface" aria-hidden="true" style="display:none"></span>', unsafe_allow_html=True)
            with st.container(height=460, border=False, key="results_content"):
                if test_mode and state.get("result", {}).get("preview_source"):
                    st.caption(f"Saved run preview: {state.result['preview_source']} · No new analysis was run.")
                report = root / "report.html" if root else None
                if report and report.is_file():
                    from aso_agents.gui.artifacts import report_url
                    st.iframe(report_url(str(root)), height="stretch")
                else:
                    st.info("The HTML report will appear when available.")
        with files:
            st.markdown('<span class="aso-output-surface" aria-hidden="true" style="display:none"></span>', unsafe_allow_html=True)
            with st.container(height=460, border=False, key="files_content"):
                if test_mode and state.get("result", {}).get("preview_source"):
                    st.caption(f"Downloads are copies of saved outputs from {state.result['preview_source']}.")
                if root and root.is_dir():
                    # List run outputs; deferred downloads avoid preloading every file.
                    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.resolve().is_relative_to(root)
                                   and not any(part.startswith(".") for part in p.relative_to(root).parts))
                    if len(paths) > 50:
                        query = st.text_input("Find a file", key=f"file_query_{root}").strip().casefold()
                        matching = [p for p in paths if query in str(p.relative_to(root)).casefold()]
                        pages = max(1, (len(matching) + 49) // 50)
                        page = st.selectbox("File page", range(1, pages + 1), key=f"file_page_{root}_{query}")
                        st.caption(f"{len(matching):,} matching files · Up to 50 per page")
                        paths = matching[(page - 1) * 50:page * 50]
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
                            with st.expander(f"Preview {relative}", expanded=path.name == "recommended_shortlist.md"):
                                if path.stat().st_size <= 1024 * 1024:
                                    with st.container(height=300, border=False):
                                        st.markdown(path.read_text(encoding="utf-8"))
                                else:
                                    st.caption("Download this file to read the full Markdown; it exceeds the 1 MB preview limit.")
                    if not paths:
                        st.info("No downloadable output files are available yet.")
                    st.caption("Files are read when you click Download. Keep the complete run folder for relative report links.")
                else:
                    st.info("Saved outputs will appear here.")

    workspace()
    st.iframe(STARTUP_READY, height=1)


if __name__ == "__main__":
    main()
