"""Loading overlay and browser readiness handshake."""
STARTUP_HTML = """<style>
.aso-startup {position:fixed;inset:0;z-index:999990;display:grid;place-items:center;
background:var(--background-color, #F5F8FC);color:#253D4B;transition:opacity .18s ease-out,visibility .18s;}
.aso-startup.aso-ready {opacity:0;visibility:hidden;pointer-events:none;}
.aso-startup-card {width:min(94vw,820px);text-align:center;font-family:Inter,sans-serif;}
.aso-startup-title {font-size:1.6rem;font-weight:900;letter-spacing:-.04em;}
.aso-startup-subtitle {font-size:clamp(.4rem,1.8vw,.95rem);font-weight:500;line-height:1.5;margin:8px auto 0;white-space:nowrap;}
.aso-rna-scene {display:block;width:100%;max-width:430px;height:auto;margin:16px auto;overflow:visible;}
.aso-rna-left,.aso-rna-right {stroke:#10A6BF;stroke-width:10;stroke-linecap:round;fill:none;transform-origin:215px 116px;}
.aso-rna-left {animation:aso-cleave-left 5.6s ease-in-out infinite;}
.aso-rna-right {animation:aso-cleave-right 5.6s ease-in-out infinite;}
.aso-gapmer {animation:aso-bind 5.6s ease-in-out infinite;}
.aso-enzyme {animation:aso-recruit 5.6s ease-in-out infinite;}
@keyframes aso-bind {
0%,12% {opacity:0;transform:translateY(-40px);}
30%,69% {opacity:1;transform:translateY(0);}
84%,100% {opacity:0;transform:translateY(-12px);}}
@keyframes aso-recruit {
0%,33% {opacity:0;transform:translate(95px,40px);}
51%,69% {opacity:1;transform:translate(0,0);}
84%,100% {opacity:0;transform:translate(0,24px);}}
@keyframes aso-cleave-left {
0%,64% {opacity:1;transform:translate(0,0) rotate(0);}
84%,91% {opacity:0;transform:translate(-42px,-24px) rotate(-18deg);}
92% {opacity:0;transform:translate(0,0) rotate(0);}
100% {opacity:1;transform:translate(0,0) rotate(0);}}
@keyframes aso-cleave-right {
0%,64% {opacity:1;transform:translate(0,0) rotate(0);}
84%,91% {opacity:0;transform:translate(42px,-24px) rotate(18deg);}
92% {opacity:0;transform:translate(0,0) rotate(0);}
100% {opacity:1;transform:translate(0,0) rotate(0);}}
.aso-startup-caption {font-size:.9rem;opacity:.75;}

@media (prefers-reduced-motion:reduce) {
.aso-rna-scene * {animation:none;}
.aso-startup {transition:none;}}
</style>
<div class="aso-startup" role="status" aria-label="Loading MODAL-ASO">
<div class="aso-startup-card"><div class="aso-startup-title">MODAL-ASO</div>
<div class="aso-startup-subtitle">Multi-mechanism Orchestration and Design by an Agentic LLM for ASOs</div>
<svg class="aso-rna-scene" viewBox="0 0 430 200" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
<path class="aso-rna-left" d="M40 116 H215"/>
<path class="aso-rna-right" d="M215 116 H390"/>
<g class="aso-gapmer" stroke="#54565A" stroke-linecap="round">
<path d="M130 110V91 M147 110V91 M164 110V91 M181 110V91 M198 110V91 M215 110V91 M232 110V91 M249 110V91 M266 110V91 M283 110V91 M300 110V91" stroke-width="4"/>
<path d="M126 88 H304" stroke-width="10"/>
</g>
<g class="aso-enzyme">
<path d="M172 116 C192 114 211 107 227 92 C238 79 244 76 252 87 C260 98 270 117 267 125 C265 134 250 130 234 137 C216 144 205 154 195 165 C187 175 179 168 172 157 C163 142 153 122 172 116Z" fill="#D0E6EE" stroke="#253D4B" stroke-width="2.5"/>
</g></svg>
<div class="aso-startup-caption">Loading your workspace…</div></div></div>"""

STARTUP_READY = """<script>
(async () => {
  const host = window.parent;
  const doc = host.document;
  host.__asoMotionCleanup?.();
  delete host.__asoMotionCleanup;
  doc.querySelectorAll('[data-testid="stElementContainer"], .element-container, .aso-landing-intro > h1, .aso-landing-intro > p, .aso-landing-intro > picture, .aso-scroll-cue')
    .forEach(el => el.style.removeProperty('translate'));
  await doc.fonts.ready;
  await Promise.all([...doc.querySelectorAll('.aso-landing-intro img')]
    .map(img => img.decode().catch(() => {})));
  const picture = doc.querySelector('.aso-landing-intro picture');
  if (picture) {
    const paint = new host.Image();
    const background = host.getComputedStyle(picture, '::before').backgroundImage;
    if (background.startsWith('url(')) {
      paint.src = background.slice(4, -1).replaceAll('"', '').replaceAll("'", '');
      await paint.decode().catch(() => {});
    }
  }
  host.requestAnimationFrame(() => host.requestAnimationFrame(() => {
    const view = doc.querySelector('.aso-workspace') ? 'workspace' : 'landing';
    if (host.__asoCurrentView !== view) {
      const main = doc.querySelector('[data-testid="stMain"]');
      main?.scrollTo({top:0, behavior:'instant'});
      host.__asoCurrentView = view;
    }
    host.__asoLayoutCleanup?.();
    if (view === 'workspace') {
      let frame = 0;
      const fitChat = () => {
        frame = 0;
        const main = doc.querySelector('[data-testid="stMain"]');
        const viewport = host.visualViewport;
        const bottom = viewport ? viewport.height + viewport.offsetTop : host.innerHeight;
        for (const [selector, property] of [
          ['.st-key-chat_history', '--aso-chat-height'],
          ['.st-key-results_content', '--aso-results-height'],
          ['.st-key-files_content', '--aso-files-height']
        ]) {
          const content = doc.querySelector(selector);
          const panel = content?.closest('[role="tabpanel"]');
          if (!content || !panel || !content.getClientRects().length) continue;
          const rect = content.getBoundingClientRect();
          const panelBottom = panel.getBoundingClientRect().bottom;
          const topAtRest = rect.top + (main?.scrollTop || 0);
          const controls = Math.max(0, panelBottom - rect.bottom);
          const height = Math.max(48, Math.floor(bottom - topAtRest - controls - 24));
          if (Math.abs(rect.height - height) > 1) {
            doc.documentElement.style.setProperty(property, height + 'px');
          }
        }
      };
      const schedule = () => { if (!frame) frame = host.requestAnimationFrame(fitChat); };
      const observer = new host.ResizeObserver(schedule);
      const content = doc.querySelector('[data-testid="stMainBlockContainer"]');
      if (content) observer.observe(content);
      doc.querySelectorAll('[data-testid="stChatInput"]').forEach(el => observer.observe(el));
      host.addEventListener('resize', schedule);
      host.visualViewport?.addEventListener('resize', schedule);
      doc.addEventListener('click', schedule);
      fitChat();
      host.__asoLayoutCleanup = () => {
        observer.disconnect();
        host.removeEventListener('resize', schedule);
        host.visualViewport?.removeEventListener('resize', schedule);
        doc.removeEventListener('click', schedule);
        if (frame) host.cancelAnimationFrame(frame);
      };
    }
    doc.documentElement.classList.add('aso-page-ready');
    doc.querySelector('.aso-startup')?.classList.add('aso-ready');
  }));
})();
</script>"""
