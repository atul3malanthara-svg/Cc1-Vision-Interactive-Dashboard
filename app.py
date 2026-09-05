"""
CC1 Vision - Streamlit dashboard.

Vision-Based Floor Stain Detection and Cleaning Decision Support
for Autonomous Cleaning Robots.

Run it with:

    python -m streamlit run app.py

This file is only the user interface. All image processing lives in
``detector.py``, all robot logic in ``cleaning_engine.py`` and
``route_simulator.py``, and all IO/report helpers in ``utils.py``.

This is an academic simulation inspired by autonomous floor-cleaning robots.
It is not affiliated with, endorsed by, or connected to any robot manufacturer.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import utils
from cleaning_engine import (
    CleaningPlan,
    build_cleaning_plans,
    plans_by_id,
    plans_to_records,
    summarize_mission,
)
from detector import (
    DetectionResult,
    DetectorConfig,
    draw_detections,
    render_mask_view,
)
from ml_classifier import get_classifier
from route_simulator import (
    DEFAULT_ROBOT_SPEED_MPS,
    DEFAULT_ZONE_WIDTH_METERS,
    plan_route,
    render_floor_map,
    route_to_records,
)
from sample_generator import generate_floor_image
from utils import ImageLoadError, load_image_from_bytes

PAGE_TITLE = "CC1 Vision - AI-Based Virtual Stain Detection"
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]
ASSET_DIR = Path(__file__).resolve().parent / "assets"

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
:root {
  --cc1-ink:#11161D; --cc1-ink-2:#1D2630; --cc1-paper:#F5F7F8; --cc1-panel:#FFFFFF;
  --cc1-line:#D9E0E5; --cc1-muted:#66717D; --cc1-blue:#61B7FF; --cc1-blue-deep:#0079E6;
  --cc1-lime:#B8FF27; --cc1-high:#E24A3B; --cc1-med:#F4A62A; --cc1-low:#2FB27C;
  --cc1-shadow:0 18px 45px rgba(18,31,44,.10);
}
html { scroll-behavior:smooth; }
.stApp { background:linear-gradient(180deg,#F8FAFB 0%,#F1F4F6 100%); color:var(--cc1-ink); }
.block-container { padding-top:1.05rem; padding-bottom:3rem; max-width:1540px; }
[data-testid="stSidebar"] { background:#11161D; border-right:1px solid #28323D; }
[data-testid="stSidebar"] * { color:#EEF3F7; }
[data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stSelectSlider label { color:#D8E1E8!important; }
[data-testid="stSidebar"] hr { border-color:#2B3641; }
.cc1-hero { position:relative; overflow:hidden; min-height:392px; border-radius:22px; padding:34px 38px;
  background:radial-gradient(circle at 76% 48%,rgba(35,126,194,.28),transparent 28%),
             radial-gradient(circle at 70% 42%,rgba(184,255,39,.10),transparent 34%),
             linear-gradient(118deg,#11161D 0%,#151E28 58%,#0E141A 100%);
  box-shadow:0 28px 65px rgba(8,17,25,.20); border:1px solid #26313B; margin-bottom:18px; }
.cc1-hero:before { content:""; position:absolute; inset:0; pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
  background-size:42px 42px; mask-image:linear-gradient(90deg,#000 30%,transparent 90%); }
.cc1-hero-copy { position:relative; z-index:4; width:52%; padding-top:10px; }
.cc1-kicker { display:inline-flex; align-items:center; gap:9px; font-size:11px; font-weight:800; letter-spacing:2.5px; color:#B9C7D2; text-transform:uppercase; }
.cc1-live-dot { width:8px;height:8px;border-radius:50%;background:var(--cc1-lime);box-shadow:0 0 0 0 rgba(184,255,39,.55);animation:pulseDot 1.8s infinite; }
.cc1-hero h1 { margin:14px 0 6px; color:white; font-size:52px; line-height:.98; letter-spacing:4px; font-weight:850; }
.cc1-hero h1 span { color:#75C8FF; }
.cc1-hero-sub { color:#AFC0CE; font-size:16px; max-width:600px; line-height:1.55; margin:0; }
.cc1-hero-flow { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:24px; }
.cc1-flow-pill { border:1px solid #334454; background:rgba(255,255,255,.045); color:#E8F0F5; padding:8px 11px; border-radius:999px; font-size:11px; font-weight:750; letter-spacing:.7px; transition:.25s ease; }
.cc1-flow-pill:hover { transform:translateY(-2px); border-color:#5BAEE7; background:rgba(83,174,234,.12); }
.cc1-flow-arrow { color:#63BFFF; font-weight:900; }
.cc1-status-chip { display:inline-flex; gap:8px; align-items:center; margin-top:18px; padding:8px 12px; border-radius:10px; font-size:11px; letter-spacing:1.1px; font-weight:800; border:1px solid rgba(255,255,255,.15); color:#F5FAFC; }
.cc1-status-chip svg { width:16px;height:16px;stroke:currentColor;fill:none; }
.cc1-status-chip.ready { background:rgba(47,178,124,.16); border-color:rgba(47,178,124,.55); }
.cc1-status-chip.alert { background:rgba(226,74,59,.17); border-color:rgba(226,74,59,.60); }
.cc1-status-chip.idle { background:rgba(255,255,255,.05); }
.cc1-robot-zone { position:absolute; right:2.2%; top:0; width:47%; height:100%; z-index:3; }
.cc1-robot { position:absolute; width:min(58%,410px); right:16%; bottom:-6px; filter:drop-shadow(0 30px 30px rgba(0,0,0,.36)); animation:robotFloat 4.8s ease-in-out infinite; z-index:4; }
.cc1-scan-ring,.cc1-scan-ring2 { position:absolute; border:1px solid rgba(104,196,255,.50); border-radius:50%; left:50%; bottom:18px; transform:translateX(-50%); width:58%; height:16%; box-shadow:0 0 28px rgba(54,162,235,.18); animation:ringPulse 2.6s ease-out infinite; }
.cc1-scan-ring2 { animation-delay:1.25s; width:44%; }
.cc1-orbit { position:absolute; left:50%; top:49%; width:76%; height:72%; transform:translate(-50%,-50%); border:1px dashed rgba(106,190,244,.24); border-radius:50%; animation:orbitTurn 18s linear infinite; }
.cc1-orbit-node { position:absolute; width:64px;height:64px;border-radius:18px;background:rgba(20,31,42,.88);border:1px solid rgba(105,192,251,.42); display:flex;align-items:center;justify-content:center;box-shadow:0 12px 28px rgba(0,0,0,.22),0 0 18px rgba(79,173,236,.10);backdrop-filter:blur(7px); }
.cc1-orbit-node svg { width:27px;height:27px;stroke:#75C8FF;fill:none;stroke-width:1.8; }
.cc1-orbit-node.n1 { left:1%;top:16%; animation:iconFloat 3.5s ease-in-out infinite; }
.cc1-orbit-node.n2 { right:1%;top:18%; animation:iconFloat 3.9s ease-in-out infinite .35s; }
.cc1-orbit-node.n3 { right:-1%;bottom:16%; animation:iconFloat 3.4s ease-in-out infinite .7s; }
.cc1-orbit-node.n4 { left:3%;bottom:12%; animation:iconFloat 3.8s ease-in-out infinite 1s; }
.cc1-sweep { position:absolute; height:2px; width:42%; background:linear-gradient(90deg,transparent,#7ACBFF,transparent); left:29%;top:51%; box-shadow:0 0 18px #56B8F8; transform-origin:left center; animation:scanner 3.2s ease-in-out infinite; z-index:2; }
.cc1-feature-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 20px; }
.cc1-feature { position:relative; overflow:hidden; background:#fff;border:1px solid var(--cc1-line);border-radius:14px;padding:17px 16px 15px;min-height:118px; transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease; animation:fadeUp .55s ease both; }
.cc1-feature:nth-child(2){animation-delay:.07s}.cc1-feature:nth-child(3){animation-delay:.14s}.cc1-feature:nth-child(4){animation-delay:.21s}
.cc1-feature:hover { transform:translateY(-5px);box-shadow:var(--cc1-shadow);border-color:#A7CFEA; }
.cc1-feature:after { content:"";position:absolute;right:-24px;bottom:-34px;width:92px;height:92px;border-radius:50%;background:rgba(97,183,255,.08); }
.cc1-icon-wrap { width:38px;height:38px;border-radius:10px;background:#11161D;display:flex;align-items:center;justify-content:center;margin-bottom:11px; animation:iconBreathe 3.3s ease-in-out infinite; }
.cc1-icon-wrap svg { width:21px;height:21px;stroke:#B8FF27;fill:none;stroke-width:1.9; }
.cc1-feature strong { display:block;font-size:13px;letter-spacing:.3px;color:#11161D;margin-bottom:4px; }
.cc1-feature span { color:#6B7681;font-size:11.5px;line-height:1.4; }
.cc1-section { display:flex;align-items:center;gap:14px;margin:7px 0 14px;padding:12px 0;border-bottom:1px solid var(--cc1-line); }
.cc1-section-num { font-size:30px;font-weight:900;color:#D4DBE1;line-height:1;letter-spacing:-1px; }
.cc1-section-icon { width:34px;height:34px;border-radius:10px;background:#11161D;display:flex;align-items:center;justify-content:center; }
.cc1-section-icon svg { width:19px;height:19px;stroke:#B8FF27;fill:none;stroke-width:1.9; }
.cc1-section h2 { margin:0;font-size:19px;color:#11161D;line-height:1.1; }
.cc1-section p { margin:4px 0 0;color:#71808D;font-size:12px; }
.cc1-metric { position:relative;overflow:hidden;background:#fff;border:1px solid var(--cc1-line);border-radius:14px;padding:15px 16px;height:100%;min-height:120px; box-shadow:0 5px 16px rgba(18,31,44,.035);transition:.25s ease;animation:fadeUp .45s ease both; }
.cc1-metric:hover { transform:translateY(-4px);box-shadow:0 16px 35px rgba(18,31,44,.09);border-color:#B9D8EC; }
.cc1-metric .topline { display:flex;justify-content:space-between;align-items:center;gap:8px; }
.cc1-metric .metric-icon { width:31px;height:31px;border-radius:9px;background:#EEF6FC;display:flex;align-items:center;justify-content:center; animation:iconBreathe 3.8s ease-in-out infinite; }
.cc1-metric .metric-icon svg { width:17px;height:17px;stroke:#087DCD;fill:none;stroke-width:2; }
.cc1-metric .label { color:#67737F;font-size:10px;letter-spacing:1.25px;text-transform:uppercase;font-weight:800; }
.cc1-metric .value { color:#11161D;font-size:27px;font-weight:850;margin-top:8px;line-height:1.04; }
.cc1-metric .note { color:#7A8792;font-size:11.5px;margin-top:6px; }
.cc1-metric.accent:before,.cc1-metric.high:before { content:"";position:absolute;left:0;top:0;right:0;height:3px;background:#2B9FEF; }.cc1-metric.high:before { background:#E24A3B; }
.cc1-panel { background:#fff;border:1px solid var(--cc1-line);border-radius:14px;padding:16px 18px;margin-bottom:14px;box-shadow:0 5px 16px rgba(18,31,44,.035); }
.cc1-panel h3 { margin:0 0 6px;font-size:15px;color:#11161D; }.cc1-panel p{margin:0;color:#6A7783;font-size:13px;line-height:1.5}.cc1-panel svg{width:18px;height:18px;stroke:#188FD4;fill:none;vertical-align:-4px;margin-right:5px}
.cc1-task { background:#fff;border:1px solid var(--cc1-line);border-left:4px solid #65727E;border-radius:13px;padding:14px 15px;margin-bottom:10px;transition:.22s ease; }.cc1-task:hover { transform:translateX(3px);box-shadow:0 10px 25px rgba(18,31,44,.08); }.cc1-task.HIGH{border-left-color:#E24A3B}.cc1-task.MEDIUM{border-left-color:#F4A62A}.cc1-task.LOW{border-left-color:#2FB27C}
.cc1-task .title{font-weight:800;color:#11161D;font-size:14px}.cc1-task .row{color:#67737F;font-size:12.5px;margin-top:4px}.cc1-task .tag{float:right;font-size:9px;letter-spacing:1.2px;padding:4px 8px;border-radius:999px;color:#fff;font-weight:900}.cc1-task .tag.HIGH{background:#E24A3B}.cc1-task .tag.MEDIUM{background:#F4A62A}.cc1-task .tag.LOW{background:#2FB27C}
.cc1-note { background:#F1F7FB;border:1px solid #CFE2EF;border-left:4px solid #339FDE;border-radius:12px;padding:13px 15px;color:#36576E;font-size:12.5px;line-height:1.55; }.cc1-foot { color:#74808B;font-size:11px;margin-top:28px;padding:16px 0;border-top:1px solid var(--cc1-line); }
.cc1-sidebrand { background:linear-gradient(145deg,#1D2934,#10171D);border:1px solid #2F3D49;border-radius:15px;padding:15px;margin:4px 0 18px;position:relative;overflow:hidden; }.cc1-sidebrand:after{content:"";position:absolute;right:-20px;top:-20px;width:90px;height:90px;border-radius:50%;background:rgba(184,255,39,.08)}
.cc1-sidebrand .mini{color:#8EA4B5;font-size:9px;letter-spacing:2px;font-weight:800}.cc1-sidebrand .name{font-size:21px;font-weight:900;letter-spacing:2px;color:#fff;margin-top:3px}.cc1-sidebrand .state{margin-top:11px;display:inline-flex;align-items:center;gap:7px;background:rgba(184,255,39,.08);border:1px solid rgba(184,255,39,.24);padding:6px 9px;border-radius:9px;color:#DFFF9D;font-size:10px;font-weight:800;letter-spacing:.8px}
[data-testid="stSidebar"] h3 { font-size:12px!important;text-transform:uppercase;letter-spacing:1.3px;color:#91A5B5!important;margin-top:18px!important; }
.stButton>button,.stDownloadButton>button { border-radius:10px!important;border:1px solid #C8D2D9!important;background:#fff!important;color:#18212A!important;font-weight:750!important;transition:.2s ease!important; }.stButton>button:hover,.stDownloadButton>button:hover { border-color:#2A96D8!important;color:#087DCD!important;transform:translateY(-1px);box-shadow:0 8px 20px rgba(20,83,125,.10); }
[data-testid="stFileUploaderDropzone"] { border:1px dashed #9DB7C9!important;border-radius:14px!important;background:#F8FBFD!important; }[data-testid="stTabs"] button { font-weight:760!important;color:#65727D!important; }.stTabs [data-baseweb="tab-highlight"]{background:#1397E7!important;height:3px!important;border-radius:3px}[data-testid="stImage"] img { border-radius:14px;box-shadow:0 8px 24px rgba(17,31,43,.08); }
@keyframes iconBreathe{0%,100%{transform:scale(1);box-shadow:0 0 0 rgba(74,174,235,0)}50%{transform:scale(1.06);box-shadow:0 0 18px rgba(74,174,235,.10)}}@keyframes robotFloat{0%,100%{transform:translateY(0) rotate(-.2deg)}50%{transform:translateY(-10px) rotate(.4deg)}}@keyframes iconFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}@keyframes ringPulse{0%{opacity:.12;transform:translateX(-50%) scale(.70)}55%{opacity:.65}100%{opacity:0;transform:translateX(-50%) scale(1.35)}}@keyframes pulseDot{0%{box-shadow:0 0 0 0 rgba(184,255,39,.45)}70%{box-shadow:0 0 0 9px rgba(184,255,39,0)}100%{box-shadow:0 0 0 0 rgba(184,255,39,0)}}@keyframes orbitTurn{from{transform:translate(-50%,-50%) rotate(0)}to{transform:translate(-50%,-50%) rotate(360deg)}}@keyframes scanner{0%,100%{transform:rotate(-11deg);opacity:.25}50%{transform:rotate(11deg);opacity:1}}@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:1000px){.cc1-hero{min-height:520px}.cc1-hero-copy{width:100%}.cc1-robot-zone{opacity:.42;width:66%;right:-10%;top:10%}.cc1-feature-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:700px){.cc1-hero{padding:25px 22px;min-height:510px}.cc1-hero h1{font-size:40px}.cc1-robot-zone{width:88%;right:-25%;top:19%}.cc1-feature-grid{grid-template-columns:1fr 1fr}.cc1-orbit-node{width:52px;height:52px}}
</style>
"""



def _asset_data_uri(filename: str) -> str:
    """Return a local asset as a data URI so the dashboard is self-contained."""
    path = ASSET_DIR / filename
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = "image/webp" if suffix == ".webp" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def ui_icon(name: str) -> str:
    """Small dependency-free line icons used by the animated product UI."""
    paths = {
        "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2V6M12 18v4M2 12h4M18 12h4"/>',
        "brain": '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 9h6v6H9zM3 9h2M3 15h2M19 9h2M19 15h2M9 3v2M15 3v2M9 19v2M15 19v2"/>',
        "route": '<circle cx="6" cy="17" r="2"/><circle cx="18" cy="7" r="2"/><path d="M8 17h3c5 0 1-10 5-10"/><path d="M16 7h-3"/>',
        "droplet": '<path d="M12 3s6 6.1 6 11a6 6 0 1 1-12 0c0-4.9 6-11 6-11z"/><path d="M9.5 16.5c1 .8 2.1 1 3.2.7"/>',
        "brush": '<path d="M8 4h8v5H8zM7 9h10v4H7zM8 13l-2 7M11 13l-1 7M14 13l1 7M17 13l2 7"/>',
        "chart": '<path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/>',
        "shield": '<path d="M12 3l7 3v5c0 4.5-2.9 7.7-7 10-4.1-2.3-7-5.5-7-10V6l7-3z"/><path d="m9 12 2 2 4-5"/>',
        "battery": '<rect x="3" y="7" width="16" height="10" rx="2"/><path d="M21 10v4M6 10v4M10 10v4M14 10v4"/>',
        "camera": '<rect x="3" y="6" width="18" height="13" rx="3"/><circle cx="12" cy="12.5" r="3.5"/><path d="M8 6l1.5-2h5L16 6"/>',
        "upload": '<path d="M12 16V4m0 0-4 4m4-4 4 4"/><path d="M4 15v4h16v-4"/>',
        "model": '<path d="M4 7l8-4 8 4-8 4-8-4zM4 12l8 4 8-4M4 17l8 4 8-4"/>',
        "alert": '<path d="M12 3 2.8 20h18.4L12 3z"/><path d="M12 9v5M12 17h.01"/>',
        "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
        "report": '<path d="M6 3h9l3 3v15H6z"/><path d="M15 3v4h4M9 11h6M9 15h6"/>',
    }
    body = paths.get(name, paths["target"])
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{body}</svg>'


def section_header(number: str, title: str, subtitle: str, icon: str) -> None:
    st.markdown(
        f'<div class="cc1-section"><div class="cc1-section-num">{number}</div>'
        f'<div class="cc1-section-icon">{ui_icon(icon)}</div><div><h2>{title}</h2>'
        f'<p>{subtitle}</p></div></div>', unsafe_allow_html=True,
    )


def render_feature_rail() -> None:
    features = [
        ("camera", "AI Perception", "OpenCV proposes visual anomalies from the floor surface."),
        ("brain", "CNN Verification", "The local CNN accepts or rejects candidates and assigns a class."),
        ("brush", "Cleaning Decision", "Severity is translated into cleaning mode, passes and priority."),
        ("route", "Route Intelligence", "Targets are ordered into a simulated autonomous cleaning route."),
    ]
    cards = "".join(
        f'<div class="cc1-feature"><div class="cc1-icon-wrap">{ui_icon(icon)}</div>'
        f'<strong>{title}</strong><span>{text}</span></div>' for icon,title,text in features
    )
    st.markdown(f'<div class="cc1-feature-grid">{cards}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Streamlit version compatibility helpers
# --------------------------------------------------------------------------


def show_image(image_rgb: np.ndarray, caption: Optional[str] = None) -> None:
    """Display an RGB image, working across Streamlit versions."""
    try:
        st.image(image_rgb, caption=caption, use_container_width=True)
    except TypeError:  # older Streamlit releases
        st.image(image_rgb, caption=caption, use_column_width=True)


def show_dataframe(frame: pd.DataFrame) -> None:
    """Display a dataframe, working across Streamlit versions."""
    try:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(frame)


# --------------------------------------------------------------------------
# Analysis (cached)
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def analyse_image(
    image_bytes: bytes,
    sensitivity: float,
    min_area_px: int,
    max_dimension: int,
    min_confidence: float,
    suppress_grout: bool,
    suppress_shadows: bool,
    suppress_glare: bool,
    denoise: bool,
    use_cnn: bool,
    min_stain_probability: float,
    model_ready: bool,
) -> DetectionResult:
    """Decode and analyse an image. Cached on the bytes plus all settings.

    ``model_ready`` is part of the cache key only so that results computed
    before the CNN was available are not served afterwards.
    """
    from detector import detect_stains  # local import keeps the cache key clean

    image = load_image_from_bytes(image_bytes)
    config = DetectorConfig(
        sensitivity=sensitivity,
        min_area_px=min_area_px,
        max_dimension=max_dimension,
        min_confidence=min_confidence,
        suppress_grout=suppress_grout,
        suppress_shadows=suppress_shadows,
        suppress_glare=suppress_glare,
        denoise=denoise,
        use_cnn=use_cnn,
        min_stain_probability=min_stain_probability,
    )
    return detect_stains(image, config, classifier=get_classifier())


@st.cache_data(show_spinner=False)
def build_sample_bytes(seed: int = 7) -> bytes:
    """Generate the synthetic demo floor and return it as JPEG bytes."""
    image, _ = generate_floor_image(seed=seed)
    return utils.encode_jpeg(image, quality=94)


# --------------------------------------------------------------------------
# Layout blocks
# --------------------------------------------------------------------------


def render_header(status_text: str, status_kind: str) -> None:
    """Draw an animated product-style hero with the CC1 Pro brochure asset."""
    robot = _asset_data_uri("cc1_pro_robot.webp")
    status_icon = "alert" if status_kind == "alert" else ("check" if status_kind == "ready" else "model")
    html = f"""<div class="cc1-hero">
      <div class="cc1-hero-copy">
        <div class="cc1-kicker"><span class="cc1-live-dot"></span> LOCAL AI CLEANING INTELLIGENCE</div>
        <h1>CC1 <span>VISION</span></h1>
        <p class="cc1-hero-sub">ML-based visual stain detection and cleaning decision support inspired by autonomous commercial cleaning workflows.</p>
        <div class="cc1-hero-flow">
          <span class="cc1-flow-pill">PERCEIVE</span><span class="cc1-flow-arrow">&rarr;</span>
          <span class="cc1-flow-pill">VERIFY</span><span class="cc1-flow-arrow">&rarr;</span>
          <span class="cc1-flow-pill">DECIDE</span><span class="cc1-flow-arrow">&rarr;</span>
          <span class="cc1-flow-pill">ROUTE</span>
        </div>
        <div class="cc1-status-chip {status_kind}">{ui_icon(status_icon)} {status_text}</div>
      </div>
      <div class="cc1-robot-zone">
        <div class="cc1-orbit"></div><div class="cc1-sweep"></div>
        <div class="cc1-orbit-node n1">{ui_icon("brain")}</div>
        <div class="cc1-orbit-node n2">{ui_icon("camera")}</div>
        <div class="cc1-orbit-node n3">{ui_icon("route")}</div>
        <div class="cc1-orbit-node n4">{ui_icon("droplet")}</div>
        <div class="cc1-scan-ring"></div><div class="cc1-scan-ring2"></div>
        <img class="cc1-robot" src="{robot}" alt="CC1 Pro cleaning robot reference image" />
      </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)
    render_feature_rail()


def metric_card(label: str, value: str, note: str = "", variant: str = "") -> str:
    """Build one animated icon metric card."""
    key = label.lower()
    icon = "chart"
    if "stain" in key: icon = "target"
    elif "confidence" in key: icon = "brain"
    elif "severity" in key or "priority" in key: icon = "alert"
    elif "coverage" in key: icon = "droplet"
    elif "status" in key: icon = "check"
    elif "water" in key: icon = "droplet"
    elif "time" in key: icon = "route"
    elif "task" in key or "target" in key: icon = "brush"
    elif "accuracy" in key or "precision" in key or "recall" in key or "f1" in key: icon = "model"
    return (
        f'<div class="cc1-metric {variant}"><div class="topline">'
        f'<div class="label">{label}</div><div class="metric-icon">{ui_icon(icon)}</div></div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>'
    )


def render_metric_strip(
    result: Optional[DetectionResult],
    mission: Optional[Dict[str, Any]],
) -> None:
    """Draw the five dashboard metrics."""
    columns = st.columns(5)

    if result is None:
        values = [
            ("Detected Stains", "--", "waiting for image"),
            ("Average Confidence", "--", "waiting for image"),
            ("High Severity Areas", "--", "waiting for image"),
            ("Dirty Floor Coverage", "--", "waiting for image"),
            ("System Status", "STANDBY", "no image loaded"),
        ]
        variants = ["", "", "", "", ""]
    else:
        high = result.high_severity_count
        if result.count == 0:
            status, note = "FLOOR CLEAN", "no action required"
        elif high > 0:
            status, note = "ACTION NEEDED", f"{high} high severity area(s)"
        else:
            status, note = "ANALYSIS COMPLETE", "routine cleaning suggested"

        mode = (mission or {}).get("recommended_mode", "-")
        values = [
            ("Detected Stains", str(result.count), f"{result.elapsed_seconds:.2f} s analysis"),
            ("Average Confidence", utils.format_percent(result.average_confidence, 0),
             result.confidence_label),
            ("High Severity Areas", str(high), "priority targets"),
            ("Dirty Floor Coverage", utils.format_percent(result.coverage_ratio, 2),
             "of the analysed image"),
            ("System Status", status, note if result.count else mode),
        ]
        variants = ["accent", "", "high" if high else "", "", "accent"]

    for column, (label, value, note), variant in zip(columns, values, variants):
        with column:
            st.markdown(metric_card(label, value, note, variant), unsafe_allow_html=True)


def render_sidebar(classifier: Any) -> Dict[str, Any]:
    """Draw the sidebar controls and return the selected settings."""
    model_ready = classifier.is_ready
    with st.sidebar:
        model_state = "CNN MODEL LOADED" if model_ready else "FALLBACK MODE"
        st.markdown(
            f'<div class="cc1-sidebrand"><div class="mini">AUTONOMOUS CLEANING CONSOLE</div>'
            f'<div class="name">CC1 VISION</div><div class="state"><span class="cc1-live-dot"></span>{model_state}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Detection controls")

        sensitivity = st.slider(
            "Detection Sensitivity", min_value=0, max_value=100, value=50, step=1,
            help="Higher values flag fainter marks and produce more false positives.",
        )
        min_area = st.slider(
            "Minimum Detection Area (px)", min_value=50, max_value=5000, value=400, step=50,
            help="Regions smaller than this are ignored. Measured on the analysis image.",
        )

        st.markdown("### Display")
        show_mask = st.checkbox("Show Detection Mask", value=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        show_contours = st.checkbox("Show Contours", value=True)
        show_confidence = st.checkbox("Show Confidence", value=True)

        with st.expander("Advanced filtering"):
            min_confidence = st.slider(
                "Minimum Confidence", min_value=0.20, max_value=0.90, value=0.35, step=0.05,
                help="Detections scoring below this are discarded.",
            )
            max_dimension = st.select_slider(
                "Analysis Resolution (longest side)",
                options=[640, 960, 1280, 1600, 2000], value=1280,
                help="Large uploads are downscaled to this size before analysis.",
            )
            denoise = st.checkbox("Noise reduction", value=True)
            suppress_grout = st.checkbox("Suppress tile joints / grout lines", value=True)
            suppress_shadows = st.checkbox("Suppress soft shadows", value=True)
            suppress_glare = st.checkbox("Suppress reflections and glare", value=True)

            use_cnn = st.checkbox(
                "Use trained CNN classifier", value=True, disabled=not model_ready,
                help=(
                    "The CNN makes the final stain / not-a-stain decision and "
                    "supplies the label and confidence."
                    if model_ready else
                    "No trained model is loaded, so the heuristic fallback is used."
                ),
            )
            min_stain_probability = st.slider(
                "Minimum CNN stain probability", min_value=0.30, max_value=0.95,
                value=0.60, step=0.05, disabled=not model_ready,
                help="Candidates the CNN scores below this are not reported as stains.",
            )

        with st.expander("Floor map settings"):
            zone_width = st.slider(
                "Assumed zone width (m)", min_value=1.0, max_value=12.0,
                value=float(DEFAULT_ZONE_WIDTH_METERS), step=0.5,
                help="Used to convert pixel distances into metres for the route.",
            )
            robot_speed = st.slider(
                "Robot travel speed (m/s)", min_value=0.2, max_value=1.5,
                value=float(DEFAULT_ROBOT_SPEED_MPS), step=0.1,
            )
            show_route = st.checkbox("Show planned route", value=True)

        st.markdown("### System information")
        import cv2  # imported here only to report the runtime version

        info_rows = [
            ("Interface", "CC1 Vision"),
            ("Version", utils.PROJECT_VERSION),
            ("Detector", "OpenCV + CNN hybrid pipeline"),
            ("Classifier",
             "TensorFlow/Keras CNN" if model_ready else "Heuristic rules (fallback)"),
            ("ML model", classifier.model_name),
            ("Model status", classifier.status),
            ("OpenCV", cv2.__version__),
            ("NumPy", np.__version__),
            ("Streamlit", st.__version__),
        ]
        if model_ready:
            config = classifier.config()
            accuracy = config.get("test_accuracy")
            info_rows.append(("Classes", str(len(classifier.classes))))
            if isinstance(accuracy, (int, float)):
                info_rows.append(("Test accuracy", f"{float(accuracy) * 100:.1f}%"))
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;font-size:12.5px;"
            f"padding:2px 0;color:#5C6B7A'><span>{key}</span>"
            f"<span style='color:#16202B'>{value}</span></div>"
            for key, value in info_rows
        )
        st.markdown(f"<div class='cc1-panel'>{rows}</div>", unsafe_allow_html=True)
        st.caption(
            "Academic concept inspired by autonomous cleaning systems. "
            "Not affiliated with any robot manufacturer."
        )

    return {
        "sensitivity": float(sensitivity),
        "min_area_px": int(min_area),
        "min_confidence": float(min_confidence),
        "max_dimension": int(max_dimension),
        "denoise": bool(denoise),
        "suppress_grout": bool(suppress_grout),
        "suppress_shadows": bool(suppress_shadows),
        "suppress_glare": bool(suppress_glare),
        "use_cnn": bool(use_cnn and model_ready),
        "min_stain_probability": float(min_stain_probability),
        "model_ready": bool(model_ready),
        "show_mask": bool(show_mask),
        "show_boxes": bool(show_boxes),
        "show_contours": bool(show_contours),
        "show_confidence": bool(show_confidence),
        "zone_width": float(zone_width),
        "robot_speed": float(robot_speed),
        "show_route": bool(show_route),
    }


# --------------------------------------------------------------------------
# Image source handling
# --------------------------------------------------------------------------


def resolve_image_source() -> Optional[Tuple[bytes, str]]:
    """Return the active image bytes and its display name, if any.

    Handles both an uploaded file and the built-in synthetic sample, and keeps
    whichever the user picked most recently.
    """
    state = st.session_state
    upload_column, sample_column, clear_column = st.columns([3, 1.1, 1.1])

    with upload_column:
        uploaded = st.file_uploader(
            "Upload a floor image (JPG, JPEG, PNG, WEBP)",
            type=UPLOAD_TYPES,
            key="uploader",
        )
    with sample_column:
        st.write("")
        if st.button("Load sample floor"):
            try:
                state["sample_bytes"] = build_sample_bytes()
                state["source"] = "sample"
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Could not generate the sample floor: {exc}")
    with clear_column:
        st.write("")
        if st.button("Clear"):
            state["source"] = None
            state["sample_bytes"] = None
            state["upload_signature"] = None

    if uploaded is not None:
        signature = (uploaded.name, getattr(uploaded, "size", 0))
        if state.get("upload_signature") != signature:
            state["upload_signature"] = signature
            state["source"] = "upload"

    if state.get("source") == "sample" and state.get("sample_bytes"):
        return state["sample_bytes"], "sample_floor.jpg (synthetic)"

    if uploaded is not None and state.get("source") != "sample":
        return uploaded.getvalue(), uploaded.name

    if state.get("sample_bytes") and state.get("source") == "sample":
        return state["sample_bytes"], "sample_floor.jpg (synthetic)"

    return None


# --------------------------------------------------------------------------
# Tab renderers
# --------------------------------------------------------------------------


def render_detection_tab(
    result: DetectionResult,
    settings: Dict[str, Any],
    annotated_bgr: np.ndarray,
    table: pd.DataFrame,
) -> None:
    """Original vs analysed image, mask, and the detection table."""
    section_header("02", "Detection Intelligence", "Compare the source image with CNN-verified visual detections.", "camera")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Original image")
        show_image(utils.to_rgb(result.image_bgr))
    with right:
        st.markdown("#### AI analysis")
        show_image(utils.to_rgb(annotated_bgr))

    if settings["show_mask"]:
        st.markdown("#### Detection mask")
        mask_column, legend_column = st.columns([3, 1])
        with mask_column:
            show_image(utils.to_rgb(render_mask_view(result)))
        with legend_column:
            st.markdown(
                "<div class='cc1-panel'><h3>Reading the mask</h3>"
                "<p>Bright areas deviate most from the surrounding floor. "
                "Only regions that passed every filter are shown at full "
                "brightness and outlined in white.</p></div>",
                unsafe_allow_html=True,
            )
            rejected = ", ".join(
                f"{key.replace('_', ' ')}: {value}"
                for key, value in result.rejected.items() if value
            ) or "none"
            st.caption(f"Filtered out - {rejected}")

    st.markdown("#### Detection table")
    if table.empty:
        st.info("No stains were detected with the current settings.")
    else:
        show_dataframe(table)
        st.caption(
            f"Coordinates are pixels on the analysis image "
            f"({result.image_bgr.shape[1]} x {result.image_bgr.shape[0]} px)."
        )


def render_recommendation_tab(plans: List[CleaningPlan], mission: Dict[str, Any]) -> None:
    """Cleaning recommendations plus the CC1 decision engine."""
    section_header("03", "Cleaning Decisions", "Turn each verified stain into a prioritized cleaning action.", "brush")
    if not plans:
        st.info("Nothing to clean - no stains were detected.")
        return

    st.markdown("#### Mission summary")
    columns = st.columns(4)
    summary_items = [
        ("Cleaning Tasks", str(mission["tasks"]), "generated from detections"),
        ("High Priority", str(mission["high_priority"]), "clean these first"),
        ("Estimated Clean Time", mission["total_clean_time_text"], "cleaning only"),
        ("Water Demand", mission["water_demand"], mission["recommended_mode"]),
    ]
    for column, (label, value, note) in zip(columns, summary_items):
        with column:
            st.markdown(metric_card(label, value, note), unsafe_allow_html=True)

    st.markdown("#### Recommended cleaning actions")
    left, right = st.columns(2)
    for index, plan in enumerate(plans):
        target = left if index % 2 == 0 else right
        with target:
            st.markdown(
                f"<div class='cc1-task {plan.priority}'>"
                f"<span class='tag {plan.priority}'>{plan.priority}</span>"
                f"<div class='title'>Stain {plan.detection_id:02d} &mdash; {plan.stain_type}</div>"
                f"<div class='row'>Recommended action: <b>{plan.action}</b></div>"
                f"<div class='row'>Cleaning intensity: {plan.intensity} "
                f"&nbsp;|&nbsp; Tool: {plan.tool}</div>"
                f"<div class='row'>Severity: {plan.severity} "
                f"&nbsp;|&nbsp; Confidence: {plan.confidence * 100:.0f}% "
                f"&nbsp;|&nbsp; Water: {plan.water_usage}</div>"
                f"<div class='row'>Passes: {plan.passes} "
                f"&nbsp;|&nbsp; Estimated time: {plan.estimated_duration_text}</div>"
                f"<div class='row'>{plan.hazard_note}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### CC1 Cleaning Decision Engine")
    st.caption("Simulated robot task queue, sorted by priority.")
    show_dataframe(pd.DataFrame(plans_to_records(plans)))


def render_map_tab(
    result: DetectionResult,
    plans: List[CleaningPlan],
    settings: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Virtual floor map with the simulated cleaning route."""
    section_header("04", "Route & Floor Map", "Simulate target ordering, travel distance and mission time.", "route")
    st.markdown("#### Virtual floor map")
    st.caption(
        "The uploaded image is treated as the cleaning zone. The robot starts at "
        "the dock and visits high priority targets first, then works outward by "
        "nearest neighbour. This is a visual simulation only."
    )

    if not plans:
        st.info("No cleaning targets to map.")
        return None

    try:
        route = plan_route(
            plans,
            result.image_bgr.shape[:2],
            zone_width_meters=settings["zone_width"],
            robot_speed_mps=settings["robot_speed"],
        )
        map_image = render_floor_map(
            result.image_bgr, route, show_route=settings["show_route"], show_labels=True
        )
    except Exception as exc:  # the map must never break the rest of the app
        st.warning(f"The floor map could not be rendered: {exc}")
        return None

    show_image(utils.to_rgb(map_image))

    summary = route.summary()
    columns = st.columns(4)
    items = [
        ("Targets", str(summary["targets"]), "cleaning stops"),
        ("Route Length", f"{summary['route_length_m']:.2f} m",
         f"zone width {summary['assumed_zone_width_m']:.1f} m"),
        ("Travel Time", summary["travel_time"], f"at {settings['robot_speed']:.1f} m/s"),
        ("Total Mission", summary["total_time"], "travel plus cleaning"),
    ]
    for column, (label, value, note) in zip(columns, items):
        with column:
            st.markdown(metric_card(label, value, note), unsafe_allow_html=True)

    st.markdown("#### Cleaning route")
    show_dataframe(pd.DataFrame(route_to_records(route)))
    return map_image


def render_reports_tab(
    result: DetectionResult,
    plans: List[CleaningPlan],
    mission: Dict[str, Any],
    settings: Dict[str, Any],
    annotated_bgr: np.ndarray,
    table: pd.DataFrame,
    source_name: str,
    map_image: Optional[np.ndarray],
) -> None:
    """Download buttons and the on-screen report preview."""
    section_header("06", "Reports", "Export visual evidence, structured detections and mission data.", "report")
    config = DetectorConfig(
        sensitivity=settings["sensitivity"],
        min_area_px=settings["min_area_px"],
        max_dimension=settings["max_dimension"],
        min_confidence=settings["min_confidence"],
        denoise=settings["denoise"],
        suppress_grout=settings["suppress_grout"],
        suppress_shadows=settings["suppress_shadows"],
        suppress_glare=settings["suppress_glare"],
        use_cnn=settings["use_cnn"],
        min_stain_probability=settings["min_stain_probability"],
    )
    summary = {**result.summary(), "mission": mission}
    report = utils.build_json_report(
        detections=result.detections,
        plans=plans,
        summary=summary,
        settings=config.as_dict(),
        image_info=utils.describe_image(result.image_bgr, source_name, result.scale),
    )

    json_bytes = utils.report_to_json_bytes(report)
    csv_bytes = utils.dataframe_to_csv_bytes(table)
    annotated_png = utils.encode_png(annotated_bgr)
    stamp = utils.timestamp_slug()

    st.markdown("#### Downloads")
    columns = st.columns(4)
    with columns[0]:
        st.download_button(
            "Annotated image (PNG)", data=annotated_png,
            file_name=f"cc1_vision_annotated_{stamp}.png", mime="image/png",
        )
    with columns[1]:
        st.download_button(
            "Detection report (JSON)", data=json_bytes,
            file_name=f"cc1_vision_report_{stamp}.json", mime="application/json",
        )
    with columns[2]:
        st.download_button(
            "Detection table (CSV)", data=csv_bytes,
            file_name=f"cc1_vision_report_{stamp}.csv", mime="text/csv",
        )
    with columns[3]:
        if map_image is not None:
            st.download_button(
                "Floor map (PNG)", data=utils.encode_png(map_image),
                file_name=f"cc1_vision_map_{stamp}.png", mime="image/png",
            )
        else:
            st.button("Floor map (PNG)", disabled=True)

    if st.button("Save all reports to the reports/ folder"):
        try:
            written = utils.save_report_bundle(annotated_png, json_bytes, csv_bytes)
            st.success("Saved:\n" + "\n".join(f"- {path}" for path in written))
        except Exception as exc:
            st.error(f"Could not write the report files: {exc}")

    st.markdown("#### JSON report preview")
    st.json(report, expanded=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def render_model_tab(classifier: Any, result: Optional[DetectionResult]) -> None:
    """Test-set performance of the locally trained CNN."""
    section_header("05", "Model Performance", "Inspect held-out synthetic-domain metrics and current-image probabilities.", "model")
    st.markdown("#### Trained model")

    if not classifier.model_exists():
        st.warning(
            "No trained model found in `models/`. The dashboard is running on the "
            "classical heuristic fallback."
        )
        st.markdown(
            "<div class='cc1-panel'><h3>Train the classifier</h3>"
            "<p>Everything runs locally on the CPU:</p></div>",
            unsafe_allow_html=True,
        )
        st.code(
            "python -m pip install -r requirements-ml.txt\n"
            "python dataset_generator.py\n"
            "python train_classifier.py",
            language="bat",
        )
        return

    if not classifier.is_ready:
        st.error(f"The model file is present but could not be loaded: {classifier.status}")
        return

    config = classifier.config()
    metrics = classifier.metrics()

    info_columns = st.columns(4)
    info_values = [
        ("Architecture", str(config.get("architecture", "CC1 lightweight CNN"))),
        ("Parameters", f"{int(config.get('parameters', 0)):,}"),
        ("Input size", f"{classifier.input_size} x {classifier.input_size} px"),
        ("Epochs trained", str(config.get("trained_epochs", "-"))),
    ]
    for column, (label, value) in zip(info_columns, info_values):
        with column:
            st.markdown(metric_card(label, value), unsafe_allow_html=True)

    st.caption(f"Framework: {classifier.framework_version()} - trained locally on the CPU.")

    if not metrics:
        st.info("No saved metrics found. Re-run `python train_classifier.py` to produce them.")
        return

    st.markdown("#### Held-out test set")
    score_columns = st.columns(5)
    scores = [
        ("Test Accuracy", utils.format_percent(metrics["test_accuracy"], 1),
         f"{metrics['test_samples']} crops", "accent"),
        ("Precision (macro)", utils.format_percent(metrics["macro_precision"], 1),
         "averaged over classes", ""),
        ("Recall (macro)", utils.format_percent(metrics["macro_recall"], 1),
         "averaged over classes", ""),
        ("F1-score (macro)", utils.format_percent(metrics["macro_f1"], 1),
         "averaged over classes", ""),
        ("Stain vs Clean F1", utils.format_percent(metrics["stain_vs_clean"]["f1"], 1),
         "the accept / reject decision", "accent"),
    ]
    for column, (label, value, note, variant) in zip(score_columns, scores):
        with column:
            st.markdown(metric_card(label, value, note, variant), unsafe_allow_html=True)

    st.markdown("#### Class-wise performance")
    per_class = metrics.get("per_class", {})
    class_rows = [
        {
            "Class": name,
            "Label": classifier.label_for(name),
            "Precision": round(values["precision"], 3),
            "Recall": round(values["recall"], 3),
            "F1-score": round(values["f1"], 3),
            "Test samples": int(values["support"]),
        }
        for name, values in per_class.items()
    ]
    if class_rows:
        show_dataframe(pd.DataFrame(class_rows))

    st.markdown("#### Confusion matrix")
    names = metrics.get("class_names", classifier.classes)
    matrix = metrics.get("confusion_matrix", [])
    if matrix:
        frame = pd.DataFrame(matrix, index=[f"true: {n}" for n in names],
                             columns=[f"pred: {n}" for n in names])
        try:
            st.dataframe(frame.style.background_gradient(cmap="Blues", axis=None),
                         use_container_width=True)
        except Exception:
            st.dataframe(frame)
        st.caption(
            "Rows are the true class, columns the predicted class. The diagonal is "
            "correct; everything off it is a confusion."
        )

    history = classifier.history()
    if history and history.get("accuracy"):
        st.markdown("#### Training curves")
        curve = pd.DataFrame({
            "train accuracy": history.get("accuracy", []),
            "validation accuracy": history.get("val_accuracy", []),
        })
        curve.index.name = "epoch"
        st.line_chart(curve)

    if result is not None and result.detections:
        st.markdown("#### CNN output for the current image")
        rows = []
        for det in result.detections:
            row = {"ID": f"{det.id:02d}", "Predicted": det.stain_type,
                   "Confidence": round(float(det.confidence), 3)}
            row.update({name: round(float(value), 3)
                        for name, value in det.ml_probabilities.items()})
            rows.append(row)
        show_dataframe(pd.DataFrame(rows))
        st.caption(
            "Per-detection softmax probabilities. The reported confidence is the "
            "probability of the predicted class."
        )

    st.markdown(
        "<div class='cc1-note'><b>How to read these numbers.</b> The model is trained "
        "and tested entirely on <b>synthetic</b> floor crops produced by "
        "<code>dataset_generator.py</code>. The scores above describe performance on "
        "that synthetic distribution only. They are not a measurement of accuracy on "
        "real photographs of real floors, and should not be read as one.</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Compose the whole dashboard."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🤖", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    classifier = get_classifier()
    settings = render_sidebar(classifier)

    source = resolve_image_source()
    if source is None:
        render_header("STANDBY", "idle")
        section_header("01", "Mission Overview", "Upload a floor image and let the local vision pipeline build a cleaning mission.", "target")
        render_metric_strip(None, None)
        st.markdown(
            f"<div class='cc1-panel'><h3>{ui_icon('upload')} Upload image to begin analysis</h3>"
            "<p>Use a real floor photo or load the synthetic sample. The system proposes anomalies with OpenCV, "
            "verifies them with the local CNN, assigns severity, recommends cleaning actions and simulates a route.</p></div>",
            unsafe_allow_html=True,
        )
        if classifier.is_ready:
            note = (
                "<b>How this works.</b> OpenCV compares every region of the floor "
                "with its own local surroundings and proposes candidate regions. A "
                "locally trained convolutional neural network then decides which "
                "candidates are real stains, which class each one is, and how "
                "confident it is. The confidence shown is the network's own "
                "probability, and the model runs entirely on this machine."
            )
        else:
            note = (
                "<b>How this works.</b> CC1 Vision compares every region of the floor "
                "with its own local surroundings using OpenCV. No trained model is "
                "loaded right now, so stain type and confidence fall back to "
                f"heuristics ({classifier.status})."
            )
        st.markdown(f"<div class='cc1-note'>{note}</div>", unsafe_allow_html=True)
        return

    image_bytes, source_name = source

    try:
        with st.spinner("Analysing floor surface..."):
            result = analyse_image(
                image_bytes,
                sensitivity=settings["sensitivity"],
                min_area_px=settings["min_area_px"],
                max_dimension=settings["max_dimension"],
                min_confidence=settings["min_confidence"],
                suppress_grout=settings["suppress_grout"],
                suppress_shadows=settings["suppress_shadows"],
                suppress_glare=settings["suppress_glare"],
                denoise=settings["denoise"],
                use_cnn=settings["use_cnn"],
                min_stain_probability=settings["min_stain_probability"],
                model_ready=settings["model_ready"],
            )
    except ImageLoadError as exc:
        render_header("IMAGE ERROR", "alert")
        st.error(f"{exc}")
        st.info("Try another file. Supported formats: JPG, JPEG, PNG, WEBP.")
        return
    except Exception as exc:  # pragma: no cover - last line of defence
        render_header("ANALYSIS FAILED", "alert")
        st.error(f"The image could not be analysed: {exc}")
        return

    plans = build_cleaning_plans(result.detections)
    mission = summarize_mission(plans, result.coverage_ratio)

    if result.count == 0:
        status, kind = "FLOOR CLEAN", "ready"
    elif result.high_severity_count:
        status, kind = "ACTION NEEDED", "alert"
    else:
        status, kind = "ANALYSIS COMPLETE", "ready"

    render_header(status, kind)
    section_header("01", "Mission Overview", "Live summary of perception, confidence, severity and cleaning status.", "target")
    render_metric_strip(result, mission)

    annotated = draw_detections(
        result.image_bgr,
        result.detections,
        show_boxes=settings["show_boxes"],
        show_contours=settings["show_contours"],
        show_labels=True,
        show_confidence=settings["show_confidence"],
    )
    table = utils.build_detection_table(result.detections, plans_by_id(plans))

    st.caption(
        f"Source: {source_name} - original {result.original_size[0]} x "
        f"{result.original_size[1]} px, analysed at "
        f"{result.image_bgr.shape[1]} x {result.image_bgr.shape[0]} px "
        f"(threshold {result.threshold_used:.2f})."
    )
    if result.uses_cnn:
        st.caption(
            f"Classification: local TensorFlow/Keras CNN "
            f"({classifier.model_name}) - {result.candidates_considered} candidate "
            f"region(s) proposed by OpenCV, {result.count} confirmed as stains. "
            f"Confidence values are CNN output probabilities."
        )
    else:
        st.caption(
            f"Classification: heuristic fallback - {result.classifier_note or classifier.status}"
        )

    detection_tab, cleaning_tab, map_tab, report_tab, model_tab = st.tabs(
        ["Detection", "Cleaning decisions", "Floor map", "Reports", "Model performance"]
    )

    with detection_tab:
        render_detection_tab(result, settings, annotated, table)
        if result.count == 0:
            st.markdown(
                "<div class='cc1-note'>No regions passed the filters. If you expected "
                "stains here, raise <b>Detection Sensitivity</b> or lower "
                "<b>Minimum Detection Area</b> in the sidebar.</div>",
                unsafe_allow_html=True,
            )

    with cleaning_tab:
        render_recommendation_tab(plans, mission)

    with map_tab:
        map_image = render_map_tab(result, plans, settings)

    with report_tab:
        render_reports_tab(
            result, plans, mission, settings, annotated, table, source_name, map_image
        )

    with model_tab:
        render_model_tab(classifier, result)

    if result.uses_cnn:
        provenance = (
            "Stain classes and confidence values are predictions from a "
            "convolutional neural network trained locally on synthetic floor "
            "crops; they describe synthetic-domain performance, not validated "
            "real-world accuracy."
        )
    else:
        provenance = (
            "No trained model is loaded, so stain categories and confidence "
            "values are heuristic estimates from classical computer vision."
        )
    st.markdown(
        "<div class='cc1-foot'>CC1 Vision - "
        "Vision-Based Floor Stain Detection and Cleaning Decision Support for "
        f"Autonomous Cleaning Robots. {provenance} Academic simulation; no physical "
        "robot is controlled.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
