import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
from engine import ADASPipeline

st.set_page_config(layout="wide", page_title="DashCAM ADAS")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stApp header { display: none; }
    .kpi-card {
        background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 8px;
        padding: 12px 16px; text-align: center;
    }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #00cc88; }
    .kpi-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
    .log-hazard { border-left: 3px solid #ff4444; background: #1a1010; padding: 6px 12px; margin: 2px 0; border-radius: 0 4px 4px 0; }
    .log-clear { border-left: 3px solid #00cc88; background: #101a10; padding: 6px 12px; margin: 2px 0; border-radius: 0 4px 4px 0; }
    .sidebar-info { font-size: 0.85rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)

st.title("DashCAM ADAS")
st.caption("Vision-Based Semantic Segmentation & Proximity Alert System")

sidebar = st.sidebar
sidebar.markdown("### System Dashboard")
sys_info = sidebar.empty()
sys_info.markdown("""
<div class="sidebar-info">
<strong>Pipeline</strong><br>
Model: SegFormer B0<br>
Dataset: Cityscapes
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload dashcam video (MP4)", type=["mp4"], label_visibility="collapsed")

if uploaded_file is None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "SegFormer B0")
    col2.metric("Dataset", "Cityscapes")
    col3.metric("Device", "CPU")
    st.info("Upload an MP4 dashcam video to start the ADAS pipeline.")
    st.stop()

with st.spinner("Loading ADAS pipeline..."):
    pipeline = ADASPipeline()

tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_file.read())
video_path = tfile.name

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / max(fps, 1)

if total_frames <= 0:
    st.error("Invalid video file.")
    os.unlink(video_path)
    st.stop()

FRAME_SKIP = 10

sys_info.markdown(f"""
<div class="sidebar-info">
<strong>Video</strong><br>
{width}x{height} | {fps:.0f} fps | {duration:.0f}s<br>
{total_frames} frames
<hr style="margin:8px 0;border-color:#2a2d3e">
<strong>Pipeline</strong><br>
Model: SegFormer B0<br>
Dataset: Cityscapes
</div>
""", unsafe_allow_html=True)

st.markdown("---")
bar = st.progress(0, text="Processing video...")

annotated_frames = []
orig_frames = []
logs = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP == 0:
        annotated, telemetry = pipeline.process_frame(frame)
        telemetry["timestamp"] = time.strftime("%H:%M:%S")
        telemetry["frame"] = frame_idx
        logs.append(telemetry)
        annotated_frames.append(annotated)
        orig_frames.append(frame)

    frame_idx += 1
    bar.progress(frame_idx / total_frames)

cap.release()
bar.empty()

play_cap = cv2.VideoCapture(video_path)

st.markdown("---")
stream_col1, stream_col2 = st.columns(2)
stream_col1.markdown("**Original Stream**")
stream_col2.markdown("**ADAS Processed Stream**")

orig_placeholder = stream_col1.empty()
proc_placeholder = stream_col2.empty()

st.markdown("---")
st.subheader("Telematics Log")
log_placeholder = st.empty()

play_bar = st.progress(0)
playback_start = time.time()
FRAME_DELAY = 1.0 / max(fps, 1)
num_processed = len(annotated_frames)

for current_frame in range(total_frames):
    ret, frame = play_cap.read()
    if not ret:
        break

    proc_idx = min(current_frame // FRAME_SKIP, num_processed - 1)
    annotated = annotated_frames[proc_idx].copy()
    telemetry_current = logs[proc_idx].copy()

    hud_frame = pipeline.draw_hud(annotated, telemetry_current, current_frame)

    rgb_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_hud = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)

    orig_placeholder.image(rgb_orig, use_container_width=True)
    proc_placeholder.image(rgb_hud, use_container_width=True)

    runtime = time.time() - playback_start
    sys_info.markdown(f"""
<div class="sidebar-info">
<strong>Video</strong><br>
{width}x{height} | {fps:.0f} fps | {duration:.0f}s<br>
{total_frames} frames
<hr style="margin:8px 0;border-color:#2a2d3e">
<strong>Pipeline</strong><br>
Model: SegFormer B0<br>
Dataset: Cityscapes
<hr style="margin:8px 0;border-color:#2a2d3e">
<strong>Live</strong><br>
Frame: {current_frame} / {total_frames}<br>
Runtime: {runtime:.1f}s / {duration:.0f}s
</div>
""", unsafe_allow_html=True)

    log_idx = min(current_frame // FRAME_SKIP, len(logs) - 1)
    visible = logs[max(0, log_idx - 24):log_idx + 1]
    log_html = ""
    for l in visible:
        if l["hazard_detected"]:
            log_html += f"""<div class="log-hazard">
                <strong style="color:#ff4444">HAZARD</strong>
                <span style="color:#888">| Frame {l['frame']} | {l['timestamp']}</span><br>
                <span style="color:#ff6666">{l['hazard_type']}</span>
                <span style="color:#888">| Occupancy: {l['warning_zone_occupancy']}%</span>
                <span style="color:#888">| V:{l['vehicle_count']} P:{l['pedestrian_count']} R:{l['road_pct']}%</span>
            </div>"""
        else:
            log_html += f"""<div class="log-clear">
                <strong style="color:#00cc88">CLEAR</strong>
                <span style="color:#888">| Frame {l['frame']} | {l['timestamp']}</span><br>
                <span style="color:#888">V:{l['vehicle_count']} P:{l['pedestrian_count']} R:{l['road_pct']}%</span>
            </div>"""
    log_placeholder.markdown(log_html, unsafe_allow_html=True)

    play_bar.progress((current_frame + 1) / total_frames)

    elapsed = time.time() - playback_start
    expected = (current_frame + 1) * FRAME_DELAY
    time.sleep(max(0.0, expected - elapsed))

play_cap.release()
play_cap = None
try:
    os.unlink(video_path)
except PermissionError:
    pass

play_bar.empty()

st.markdown("---")
st.subheader("Session Summary")

hazard_count = sum(1 for l in logs if l["hazard_detected"])
max_occ = max((l["warning_zone_occupancy"] for l in logs), default=0)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_frames}</div><div class="kpi-label">Total Frames</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="kpi-card"><div class="kpi-value">{num_processed}</div><div class="kpi-label">Inference Runs</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{"#ff4444" if hazard_count else "#00cc88"}">{hazard_count}</div><div class="kpi-label">Hazards Detected</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="kpi-card"><div class="kpi-value">{max_occ}%</div><div class="kpi-label">Max Zone Occupancy</div></div>', unsafe_allow_html=True)

if logs:
    df = pd.DataFrame(logs)
    df["display_frame"] = df["frame"]
    df["occupancy"] = df["warning_zone_occupancy"]
    df["hazard_intensity"] = df["occupancy"].where(df["hazard_detected"], 0)

    chart_data = df[["display_frame", "occupancy", "hazard_intensity"]].set_index("display_frame")
    st.markdown("#### Warning Zone Occupancy Timeline")
    st.area_chart(chart_data, height=250, color=["#555", "#ff4444"])
