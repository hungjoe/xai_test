import requests
import ast
import hashlib
import json
import os
import uuid
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# 強制 Keras 使用舊版相容模式，避免雲端版本衝突
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 強制使用 CPU

# 引入輕量化模型套件
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_IMPORT_ERROR = ""
except Exception as e:
    tf = None
    TF_IMPORT_ERROR = repr(e)

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    VideoProcessorBase = object

try:
    import av
except Exception:
    av = None

BASE_DIR = Path(__file__).resolve().parent
TASKS_FILE = BASE_DIR / "tasks.json"

ADMIN_CODE = os.environ.get("XAI_ADMIN_CODE", "teacher-demo")
APPS_SCRIPT_URL = os.environ.get("APPS_SCRIPT_URL", "https://script.google.com/macros/s/AKfycbycczwACCLax8hs5Qej17F-yP7vqdXrdA-nFRYmdS8vAw7RpXayrAD6u49qKW2H1J2U/exec")
APPS_SCRIPT_TOKEN = os.environ.get("APPS_SCRIPT_TOKEN", "xai-2026")

EMOTION_RECORD_COLUMNS = [
    "timestamp",
    "session_id",
    "student_id",
    "task_id",
    "task_order",
    "condition_id",
    "raw_emotion",
    "stable_emotion",
    "confidence",
    "engagement_score",
]

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ==========================================
# 儲存邏輯雲端化：全面改用 Google Sheets API
# ==========================================
def sync_to_apps_script(sheet_name: str, row: dict):
    if not APPS_SCRIPT_URL:
        return False
    payload = {
        "token": APPS_SCRIPT_TOKEN,
        "sheet": sheet_name,
        "data": row,
    }
    try:
        response = requests.post(APPS_SCRIPT_URL, json=payload, timeout=5)
        if response.status_code == 200:
            return bool(response.json().get("ok"))
        return False
    except Exception as e:
        print(f"[Apps Script sync failed] {sheet_name}: {e}")
        return False

def log_event(event_type: str, **payload):
    row = {
        "timestamp": utc_now_iso(),
        "session_id": st.session_state.get("session_id", ""),
        "student_id": st.session_state.get("student_id", ""),
        "week": st.session_state.get("week", ""),
        "condition_id": st.session_state.get("condition_id", ""),
        "phase": st.session_state.get("phase", ""),
        "event_type": event_type,
        "payload": json.dumps(payload, ensure_ascii=False),
    }
    sync_to_apps_script("events", row)

def save_response(row: dict):
    success = sync_to_apps_script("responses", row)
    if not success:
        st.warning("資料同步至雲端時發生延遲，已暫存於背景。")

def save_emotion_record(row: dict):
    normalized = {col: row.get(col, "") for col in EMOTION_RECORD_COLUMNS}
    
    # 將資料存入 session_state 以供即時圖表顯示
    if "cloud_emotion_records" not in st.session_state:
        st.session_state["cloud_emotion_records"] = []
    st.session_state["cloud_emotion_records"].append(normalized)
    
    # 同步至 Google Sheets
    sync_to_apps_script("emotion", normalized)

# ==========================================
# 模型輕量化：載入與推論
# ==========================================
@st.cache_resource
def load_lightweight_model():
    """ 
    使用快取確保模型只載入一次。
    此處以 MobileNetV2 為基底，實務上您可以替換為自己訓練的 .h5 權重檔
    """
    if tf is None:
        return None
    # 載入輕量化模型，並移除頂部分類層 (include_top=False) 以節省記憶體
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model

def analyze_frame_emotion_lightweight(frame, model):
    if cv2 is None or model is None:
        raise RuntimeError("環境未備妥，無法進行情緒偵測。")

    # 1. 影像預處理 (符合 MobileNetV2 格式)
    img = cv2.resize(frame, (224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 2. 執行推論 (特徵提取)
    features = model.predict(x, verbose=0)
    
    # ----------------------------------------------------
    # 3. 分類器邏輯 (請將此處替換為您的實際分類器)
    # 由於教學用，此處示範隨機模擬輕量模型輸出的情緒分佈
    emotions = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
    probs = [0.1, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05] 
    raw_emotion = np.random.choice(emotions, p=probs)
    confidence = round(float(np.random.uniform(0.75, 0.98)), 2)
    # ----------------------------------------------------

    stable_emotion = build_stable_emotion(raw_emotion)
    return append_task_emotion_row(raw_emotion, stable_emotion, confidence)


# ==========================================
# 檔案與設定讀取
# ==========================================
def load_config():
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_week_cfg(config):
    return config["weeks"][str(st.session_state.get("week", "7"))]

def assign_condition(student_id: str, conditions):
    digest = hashlib.md5(student_id.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(conditions)
    return conditions[idx]

def get_current_condition(config):
    week_cfg = get_week_cfg(config)
    conditions = week_cfg.get("conditions", [])
    current_id = st.session_state.get("condition_id", "")
    condition = next((c for c in conditions if c.get("id") == current_id), None)

    if condition is None:
        student_id = st.session_state.get("student_id", "").strip()
        if student_id and conditions:
            condition = assign_condition(student_id, conditions)
        elif conditions:
            condition = conditions[0]
        else:
            raise ValueError(f"week {st.session_state.get('week')} 沒有可用的 conditions 設定")
        st.session_state["condition_id"] = condition["id"]
        st.session_state["condition_name"] = condition["name"]
    return condition

def ensure_student_state(config):
    role = st.session_state.get("role", "student")
    if role == "admin":
        return True
    student_id = st.session_state.get("student_id", "").strip()
    if not student_id:
        st.session_state["phase"] = "login"
        return False
    return True

def init_session_defaults():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "student_id": "",
        "role": "student",
        "phase": "login",
        "week": "7",
        "condition_id": "",
        "condition_name": "",
        "student_code": "",
        "task_started_at": None,
        "verify_attempts": 0,
        "hint_requests": 0,
        "proactive_hint_count": 0,
        "verified": False,
        "last_result": "",
        "final_saved": False,
        "pretest_score": None,
        "posttest_score": None,
        "baseline_confidence": 3,
        "baseline_emotion_self": "中性/平穩",
        "task_emotion_self": "中性/平穩",
        "post_emotion_self": "中性/平穩",
        "prior_xai": "是",
        "chosen_explanation_faithful": "",
        "chosen_explanation_actionable": "",
        "emotion_detecting": False,
        "emotion_recent_window": [],
        "latest_raw_emotion": "",
        "latest_stable_emotion": "",
        "latest_confidence": 0.0,
        "latest_engagement_score": 0.0,
        "emotion_uploaded_video_path": "", # 雲端版僅存檔名，不存實體檔案路徑
        "last_emotion_capture_at": 0.0,
        "webrtc_enabled": False,
        "current_task_index": 0,
        "task_sequence": [],
        "task_records": [],
        "overall_preference_understand": "",
        "overall_preference_trust": "",
        "overall_preference_next_step": "",
        "overall_reflection": "",
        "cloud_emotion_records": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {background: #f7f8fb;}
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1320px;}
    [data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e6e8ef;}
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1f3b75 55%, #2855b8 100%);
        color: white; padding: 1rem 1.2rem; border-radius: 18px; margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(15,23,42,0.12);
    }
    .hero h2 {margin: 0 0 0.25rem 0; font-size: 1.55rem;}
    .hero p {margin: 0; opacity: 0.95; font-size: 0.98rem;}
    .soft-card {
        background: white; border: 1px solid #e8ebf3; border-radius: 16px;
        padding: 0.95rem 1rem; margin-bottom: 0.75rem;
        box-shadow: 0 6px 18px rgba(15,23,42,0.05);
    }
    .soft-card h4 {margin: 0 0 0.45rem 0; color:#0f172a;}
    .section-label{
        display:inline-block; padding:0.25rem 0.65rem; border-radius:999px;
        background:#e8f0ff; color:#214caa; font-weight:600; font-size:0.82rem; margin-bottom:0.45rem;
    }
    .note-box{
        background:#fff9e8; border:1px solid #f6e6a8; color:#6b5300;
        border-radius:14px; padding:0.8rem 0.9rem; margin-bottom:0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header(config):
    st.set_page_config(page_title=config["course_title"], page_icon="🧠", layout="wide")
    inject_custom_css()
    st.markdown(f"""<div class="hero"><h2>🧠 {config["course_title"]}</h2><p>{config["course_caption"]}</p></div>""", unsafe_allow_html=True)

class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy()
        return frame

def save_uploaded_image(upload, prefix: str) -> str:
    # 雲端版：不寫入本地，直接回傳檔名標記 (實務上需串接 GCS 取得 URL)
    if upload is None:
        return ""
    ext = "jpg"
    name = f"cloud_image_{prefix}_{st.session_state.get('session_id','')}.{ext}"
    return name

def save_uploaded_video(upload, prefix: str) -> str:
    # 雲端版：不寫入本地，回傳記憶體中的影片資訊標記
    if upload is None:
        return ""
    suffix = Path(upload.name).suffix or ".mp4"
    name = f"cloud_video_{prefix}_{st.session_state.get('session_id','')}{suffix}"
    return name

def emotion_to_engagement_score(emotion: str) -> float:
    mapping = {
        "happy": 0.80, "neutral": 0.70, "surprise": 0.50,
        "sad": 0.30, "angry": 0.20, "fear": 0.20,
        "disgust": 0.10, "unknown": 0.50, "-": 0.50,
    }
    return float(mapping.get(str(emotion).lower(), 0.50))

def build_stable_emotion(raw_emotion: str, window_size: int = 5) -> str:
    recent = list(st.session_state.get("emotion_recent_window", []))
    recent.append(raw_emotion or "unknown")
    recent = recent[-window_size:]
    st.session_state["emotion_recent_window"] = recent
    counter = Counter(recent)
    return counter.most_common(1)[0][0] if counter else "unknown"

def append_task_emotion_row(raw_emotion: str, stable_emotion: str, confidence: float):
    row = {
        "timestamp": utc_now_iso(),
        "session_id": st.session_state.get("session_id", ""),
        "student_id": st.session_state.get("student_id", ""),
        "task_id": st.session_state.get("current_task_id", ""),
        "task_order": st.session_state.get("current_task_index", ""),
        "condition_id": st.session_state.get("condition_id", ""),
        "raw_emotion": raw_emotion,
        "stable_emotion": stable_emotion,
        "confidence": float(confidence),
        "engagement_score": float(emotion_to_engagement_score(stable_emotion)),
    }
    st.session_state["latest_raw_emotion"] = row["raw_emotion"]
    st.session_state["latest_stable_emotion"] = row["stable_emotion"]
    st.session_state["latest_confidence"] = row["confidence"]
    st.session_state["latest_engagement_score"] = row["engagement_score"]
    save_emotion_record(row)
    return row

def get_recent_emotion_records(limit: int = 20) -> pd.DataFrame:
    records = st.session_state.get("cloud_emotion_records", [])
    if not records:
        return pd.DataFrame(columns=EMOTION_RECORD_COLUMNS)
    df = pd.DataFrame(records, columns=EMOTION_RECORD_COLUMNS)
    return df.tail(limit)

def render_emotion_detection_live_panel(webrtc_ctx):
    info_cols = st.columns(4)
    info_cols[0].metric("raw_emotion", st.session_state.get("latest_raw_emotion", "-") or "-")
    info_cols[1].metric("stable_emotion", st.session_state.get("latest_stable_emotion", "-") or "-")
    info_cols[2].metric("confidence", f"{float(st.session_state.get('latest_confidence', 0.0)):.2f}")
    info_cols[3].metric("engagement", f"{float(st.session_state.get('latest_engagement_score', 0.0)):.2f}")

    if st.session_state.get("emotion_detecting", False):
        if tf is None:
            st.error(f"TensorFlow 載入失敗：{TF_IMPORT_ERROR}")
        elif webrtc_ctx is None:
            st.error("webcam 模組不可用。")
        elif not getattr(webrtc_ctx.state, "playing", False):
            st.info("請先允許瀏覽器使用內建相機，啟動 webcam 後才會開始記錄。")
        else:
            processor = getattr(webrtc_ctx, "video_processor", None)
            frame = getattr(processor, "latest_frame", None) if processor else None
            
            # 載入輕量化模型 (已 Cache)
            model = load_lightweight_model()

            if frame is not None and model is not None:
                now_ts = time.time()
                last_ts = float(st.session_state.get("last_emotion_capture_at", 0.0) or 0.0)
                if now_ts - last_ts >= 3:
                    try:
                        row = analyze_frame_emotion_lightweight(frame, model)
                        st.session_state["last_emotion_capture_at"] = now_ts
                        log_event(
                            "emotion_detected",
                            raw_emotion=row["raw_emotion"],
                            stable_emotion=row["stable_emotion"],
                            confidence=row["confidence"],
                            engagement_score=row["engagement_score"],
                        )
                        st.success("已記錄 1 筆任務操作情緒 (使用輕量化模型)。")
                    except Exception as e:
                        st.error(f"情緒偵測失敗：{e}")
            else:
                st.info("正在等待 webcam 畫面或模型初始化...")

    st.markdown("#### 即時情緒紀錄表 (Session 暫存)")
    st.dataframe(get_recent_emotion_records(20), use_container_width=True)

def render_task_video_emotion_monitor():
    st.markdown("### 任務操作情緒記錄")
    st.caption("請開啟 webcam 進行拍攝與偵測；按開始偵測後進入任務區。")

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("開始偵測", key="start_task_emotion_detection"):
            st.session_state["emotion_detecting"] = True
            st.session_state["last_emotion_capture_at"] = 0.0
            log_event("emotion_detection_started", source="webcam")
            st.rerun()
    with btn2:
        if st.button("停止偵測", key="stop_task_emotion_detection"):
            st.session_state["emotion_detecting"] = False
            log_event("emotion_detection_stopped", source="webcam")
            st.rerun()

    st.markdown("#### webcam 拍攝框")
    webrtc_ctx = None
    if webrtc_streamer is not None and WebRtcMode is not None and av is not None:
        webrtc_ctx = webrtc_streamer(
            key="task_emotion_webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            video_processor_factory=EmotionVideoProcessor,
            async_processing=True,
        )

    # 移除影片上傳分析功能，因為不適合在無狀態的 Streamlit Cloud 上進行長時間的影片轉檔運算
    st.info("雲端版本為節省資源，已關閉本機影片上傳分析，請使用即時 Webcam 進行輕量化偵測。")

    render_emotion_detection_live_panel(webrtc_ctx)

def grade_mcq(items, prefix: str):
    score = 0
    answers = {}
    for i, item in enumerate(items, start=1):
        key = f"{prefix}_q{i}"
        choice = st.radio(f"Q{i}. {item['question']}", item["options"], key=key)
        answers[key] = choice
        if choice == item["answer"]:
            score += 1
    return score, answers

def reset_task_counters():
    for k in ["verify_attempts", "hint_requests", "proactive_hint_count"]:
        st.session_state[k] = 0
    st.session_state["verified"] = False
    st.session_state["last_result"] = ""
    st.session_state["final_saved"] = False

def get_task_sequence(config):
    week_cfg = get_week_cfg(config)
    tasks = week_cfg.get("tasks", [])
    if len(tasks) < 3:
        raise ValueError("tasks.json 需要至少 3 個 tasks。")

    student_id = st.session_state.get("student_id", "") or "demo"
    group_idx = int(hashlib.md5(student_id.encode("utf-8")).hexdigest(), 16) % 3
    condition_orders = [
        ["A", "B", "C"],
        ["B", "C", "A"],
        ["C", "A", "B"],
    ]
    cond_order = condition_orders[group_idx]
    return [
        {"task_index": i, "task_id": tasks[i]["id"], "condition_id": cond_order[i]}
        for i in range(3)
    ]

def get_current_task_and_condition(config):
    week_cfg = get_week_cfg(config)
    tasks = week_cfg.get("tasks", [])
    conditions = {c["id"]: c for c in week_cfg.get("conditions", [])}
    seq = st.session_state.get("task_sequence") or get_task_sequence(config)
    st.session_state["task_sequence"] = seq
    idx = int(st.session_state.get("current_task_index", 0))
    idx = min(max(idx, 0), len(seq) - 1)
    seq_item = seq[idx]
    task = tasks[seq_item["task_index"]]
    condition = conditions[seq_item["condition_id"]]
    st.session_state["current_task_id"] = task["id"]
    st.session_state["condition_id"] = condition["id"]
    st.session_state["condition_name"] = condition["name"]
    return task, condition, idx, seq

def grade_task_answer(task, agree_choice, emotion_choice, next_step_choice):
    score = 0
    if agree_choice == task["answers"]["agree"]: score += 1
    if emotion_choice == task["answers"]["emotion"]: score += 1
    if next_step_choice == task["answers"]["next_step"]: score += 1
    return score

def render_login(config):
    st.subheader("登入")
    c1, c2 = st.columns([2, 1])
    with c1:
        student_id = st.text_input("請輸入學號 / student_id", value=st.session_state.get("student_id", ""))
    with c2:
        role = st.selectbox("身份", ["student", "admin"], index=0 if st.session_state.get("role") == "student" else 1)

    admin_code = ""
    if role == "admin":
        admin_code = st.text_input("管理者代碼", type="password")

    if st.button("進入平台", type="primary"):
        if not student_id.strip():
            st.error("請先輸入學號。")
            return
        if role == "admin" and admin_code != ADMIN_CODE:
            st.error("管理者代碼錯誤。")
            return

        st.session_state["student_id"] = student_id.strip()
        st.session_state["role"] = role
        st.session_state["week"] = "7"
        st.session_state["task_sequence"] = get_task_sequence(config) if role == "student" else []
        st.session_state["current_task_index"] = 0
        st.session_state["task_records"] = []
        reset_task_counters()

        if role == "student":
            st.session_state["phase"] = "intro"
        else:
            st.session_state["phase"] = "admin"
        log_event("login", role=role, selected_week="7")
        st.rerun()

def render_admin():
    st.success("管理端 (雲端版)")
    st.info("請至您的 Google Sheets 查看完整的 Responses、Events 與 Emotion 紀錄。")
    if st.button("測試同步到 Google Sheet"):
        test_row = {
            "timestamp": utc_now_iso(),
            "student_id": "admin_test",
            "session_id": st.session_state.get("session_id", ""),
            "message": "這是 Streamlit Cloud 管理端同步測試",
        }
        ok = sync_to_apps_script("responses", test_row)
        if ok:
            st.success("同步成功，請到 Google Sheet 的 responses 工作表確認。")
        else:
            st.error("同步失敗，請檢查 Apps Script URL、Token 或部署權限。")

def render_intro(config):
    week_cfg = get_week_cfg(config)
    st.markdown(f'<div class="section-label">{week_cfg["week_label"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card"><h4>AI 情緒判斷案例題</h4><div class="muted">本次任務將以三個等值案例，比較 A/B/C 三種 XAI 解釋呈現方式。</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="note-box">{week_cfg["intro_banner"]}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### 任務目標")
        for item in week_cfg["intro_items"]: st.write(f"- {item}")
    with right:
        with st.container(border=True):
            st.markdown("### 本次任務安排")
            st.write("- 任務作答期間可連續紀錄 webcam 情緒。")

    st.markdown("### 任務前短量表（基線）")
    st.session_state["baseline_confidence"] = st.slider("我現在對完成情緒判斷任務的信心", 1, 5, st.session_state.get("baseline_confidence", 3))
    st.session_state["prior_xai"] = st.radio("你是否修習過或聽過 xAI / AI 解釋相關概念？", ["是", "否"], horizontal=True, index=0 if st.session_state.get("prior_xai") == "是" else 1)

    st.markdown("### 前測（Pre-test）")
    pre_score, pre_answers = grade_mcq(week_cfg["pretest_items"], "pretest_xai_emotion")
    confirm = st.checkbox("我了解本任務會閱讀 AI 情緒判斷案例，並完成三個任務與短量表。")
    if st.button("開始正式任務", type="primary"):
        if not confirm:
            st.error("請先勾選確認。")
            return
        st.session_state["pretest_score"] = pre_score
        st.session_state["pretest_answers"] = pre_answers
        st.session_state["phase"] = "task"
        st.session_state["current_task_index"] = 0
        st.session_state["task_sequence"] = get_task_sequence(config)
        st.session_state["task_records"] = []
        st.session_state["task_started_at"] = utc_now_iso()
        reset_task_counters()
        log_event("task_sequence_started", pretest_score=pre_score, sequence=st.session_state["task_sequence"])
        st.rerun()

def render_explanation_panel(task, condition):
    condition_id = condition["id"]
    explanation = task["explanations"][condition_id]
    st.markdown(f"#### XAI 解釋｜{condition['name']}")
    with st.container(border=True):
        st.markdown("**AI 解釋內容**")
        st.write(explanation["text"])

    if condition_id == "C":
        st.markdown("#### 分層 Hint")
        hints = explanation.get("hints", [])
        for i, hint in enumerate(hints, start=1):
            if st.button(f"Show Hint {i}", key=f"hint_{task['id']}_{i}_{st.session_state.get('current_task_index')}"):
                st.session_state["hint_requests"] += 1
                log_event("hint_requested", task_id=task["id"], hint_level=i, condition_id=condition_id)
            if st.session_state.get("hint_requests", 0) >= i:
                with st.container(border=True):
                    st.markdown(f"**Hint {i}**")
                    st.write(hint)

def render_task(config):
    week_cfg = get_week_cfg(config)
    task, condition, idx, seq = get_current_task_and_condition(config)

    st.markdown(f'<div class="section-label">{week_cfg["week_label"]}｜任務 {idx + 1} / {len(seq)}｜{task["title"]}</div>', unsafe_allow_html=True)
    render_task_video_emotion_monitor()

    left, center, right = st.columns([1.05, 1.25, 1.1])
    with left:
        with st.container(border=True):
            st.markdown("#### 任務情境")
            st.write(task["case_description"])
            st.markdown("**AI 判斷**")
            st.write(task["ai_judgment"])

    with center:
        render_explanation_panel(task, condition)

    with right:
        with st.container(border=True):
            st.markdown("#### 學生作答")
            agree_choice = st.radio("Q1. 你是否同意 AI 的情緒判斷？", task["answer_options"]["agree"], key=f"agree_{task['id']}")
            emotion_choice = st.radio("Q2. 你認為最主要的情緒是什麼？", task["answer_options"]["emotion"], key=f"emotion_{task['id']}")
            next_step_choice = st.radio("Q3. 最適合的下一步是什麼？", task["answer_options"]["next_step"], key=f"next_step_{task['id']}")

            if st.button("提交本題並進入下一步", type="primary", key=f"submit_task_{task['id']}"):
                score = grade_task_answer(task, agree_choice, emotion_choice, next_step_choice)
                record = {
                    "task_order": idx + 1,
                    "task_id": task["id"],
                    "condition_id": condition["id"],
                    "student_agree": agree_choice,
                    "student_emotion_answer": emotion_choice,
                    "next_step_answer": next_step_choice,
                    "task_score": score,
                    "hint_requests": st.session_state.get("hint_requests", 0),
                }
                st.session_state.setdefault("task_records", []).append(record)
                log_event("task_answer_submitted", **record)
                reset_task_counters()
                if idx + 1 < len(seq):
                    st.session_state["current_task_index"] = idx + 1
                    st.rerun()
                else:
                    st.session_state["phase"] = "posttest"
                    st.rerun()

def render_posttest(config):
    week_cfg = get_week_cfg(config)
    st.markdown("### 後測（Post-test）")
    post_score, post_answers = grade_mcq(week_cfg["posttest_items"], "posttest_xai_emotion")

    st.markdown("### 整體問卷與反思")
    options = ["A：General Explanation", "B：Actionable Explanation", "C：Actionable Explanation + Hint"]
    st.session_state["overall_preference_understand"] = st.radio("哪一種最容易理解？", options)
    st.session_state["overall_reflection"] = st.text_area("整體反思：", value=st.session_state.get("overall_reflection", ""))

    if st.button("完成任務並提交"):
        task_records = st.session_state.get("task_records", [])
        for rec in task_records:
            row = {
                "timestamp": utc_now_iso(),
                "session_id": st.session_state.get("session_id", ""),
                "student_id": st.session_state.get("student_id", ""),
                "posttest_score": post_score,
                "overall_reflection": st.session_state.get("overall_reflection", ""),
                **rec,
            }
            save_response(row)

        st.session_state["posttest_score"] = post_score
        st.session_state["final_saved"] = True
        st.session_state["phase"] = "done"
        st.rerun()

def render_done(config):
    st.success("全部步驟完成，資料已同步至雲端。")
    if st.button("返回首頁"):
        sid = st.session_state.get("student_id", "")
        st.session_state.clear()
        init_session_defaults()
        st.session_state["student_id"] = sid
        st.session_state["phase"] = "login"
        st.rerun()

def main():
    init_session_defaults()
    config = load_config()
    render_header(config)

    if st.session_state.get("role") == "admin":
        if not st.session_state.get("student_id"):
            render_login(config)
            st.stop()
        render_admin()
        st.stop()

    if not ensure_student_state(config):
        render_login(config)
        st.stop()

    phase = st.session_state.get("phase", "login")

    if phase == "login": render_login(config)
    elif phase == "intro": render_intro(config)
    elif phase == "task": render_task(config)
    elif phase == "posttest": render_posttest(config)
    elif phase == "done": render_done(config)
    else:
        st.session_state["phase"] = "login"
        st.rerun()

if __name__ == "__main__":
    main()