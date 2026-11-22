import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import json
import html

# =========================================================
# EduPredict AI — Student Performance Prediction Dashboard
# =========================================================
st.set_page_config(
    page_title="EduPredict AI | Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("best_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_models()
except Exception as exc:
    st.error(
        "Model files could not be loaded. Please place "
        "`best_model.pkl` and `preprocessor.pkl` in the same folder as `app.py`."
    )
    st.code(str(exc))
    st.stop()

# -----------------------------
# Session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Predictor"
if "language" not in st.session_state:
    st.session_state.language = "English"

# -----------------------------
# Localization
# -----------------------------
TRANSLATIONS = {
    "မြန်မာ": {
        "Workspace": "အလုပ်နေရာ", "Navigate": "စာမျက်နှာရွေးရန်",
        "Predictor": "ခန့်မှန်းချက်", "Analytics": "ဆန်းစစ်ချက်", "About Project": "ပရောဂျက်အကြောင်း",
        "Model Information": "မော်ဒယ်အချက်အလက်", "Prediction Engine": "ခန့်မှန်းစနစ်",
        "Task · Regression": "လုပ်ငန်း · Regression", "Dataset · 6,607 records": "ဒေတာ · မှတ်တမ်း ၆,၆၀၇ ခု",
        "Student information is used for the current prediction session.": "ကျောင်းသားအချက်အလက်ကို လက်ရှိခန့်မှန်းမှုအတွက်သာ အသုံးပြုပါသည်။",
        "Intelligent learning insights": "ဉာဏ်ရည်မြင့် သင်ယူမှုအချက်အလက်များ",
        "STUDENT SUCCESS PLATFORM": "ကျောင်းသားအောင်မြင်မှု ပလက်ဖောင်း",
        "Machine learning dashboard": "စက်လေ့လာမှု ဒက်ရှ်ဘုတ်",
        "Understand student performance through data. Enter a learner profile and generate an estimated exam score with clear, actionable insights.": "ဒေတာများမှတစ်ဆင့် ကျောင်းသား၏ စွမ်းဆောင်ရည်ကို နားလည်ပါ။ ကျောင်းသားအချက်အလက်များထည့်ပြီး ခန့်မှန်းစာမေးပွဲရမှတ်နှင့် အသုံးချနိုင်သော အကြံပြုချက်များကို ရယူပါ။",
        "Smart prediction": "ဉာဏ်ရည်မြင့် ခန့်မှန်းမှု", "Performance analytics": "စွမ်းဆောင်ရည် ဆန်းစစ်ချက်", "Student-focused": "ကျောင်းသားကို ဦးစားပေး",
        "Total predictions": "ခန့်မှန်းမှု စုစုပေါင်း", "Records in this session": "ဤ session အတွင်း မှတ်တမ်းများ",
        "Average score": "ပျမ်းမျှရမှတ်", "Across all predictions": "ခန့်မှန်းမှုအားလုံး၏ ပျမ်းမျှ",
        "Highest score": "အမြင့်ဆုံးရမှတ်", "Best predicted result": "အကောင်းဆုံး ခန့်မှန်းရလဒ်",
        "Build a student profile": "ကျောင်းသားအချက်အလက် ဖန်တီးရန်", "Complete the details below to create a prediction": "ခန့်မှန်းမှုဖန်တီးရန် အောက်ပါအချက်အလက်များကို ဖြည့်ပါ",
        "Academic": "ပညာရေး", "Lifestyle": "နေထိုင်မှုပုံစံ", "Learning environment": "သင်ယူမှုပတ်ဝန်းကျင်",
        "Academic performance": "ပညာရေး စွမ်းဆောင်ရည်", "Health, routine & personal profile": "ကျန်းမာရေး၊ နေ့စဉ်ဘဝနှင့် ကိုယ်ရေးအချက်အလက်",
        "School, family & resources": "ကျောင်း၊ မိသားစုနှင့် အရင်းအမြစ်များ",
        "Student name (optional)": "ကျောင်းသားအမည် (မဖြည့်လည်းရ)", "Generate exam score": "စာမေးပွဲရမှတ် ခန့်မှန်းရန်",
        "Performance analytics": "စွမ်းဆောင်ရည် ဆန်းစစ်ချက်", "Review predictions generated in this session": "ဤ session အတွင်း ပြုလုပ်ထားသော ခန့်မှန်းမှုများကို ကြည့်ရန်",
        "Prediction result": "ခန့်မှန်းရလဒ်", "Your latest student performance estimate": "နောက်ဆုံး ကျောင်းသားစွမ်းဆောင်ရည် ခန့်မှန်းချက်",
        "Predicted exam score": "ခန့်မှန်းစာမေးပွဲရမှတ်", "Learner profile": "ကျောင်းသားအချက်အလက်",
        "Performance indicator": "စွမ်းဆောင်ရည် အညွှန်း", "Prediction records": "ခန့်မှန်းမှု မှတ်တမ်းများ",
        "No analytics yet": "ဆန်းစစ်ချက် မရှိသေးပါ", "Create your first student prediction to see charts, averages, and records here.": "ပထမဆုံး ကျောင်းသားခန့်မှန်းမှုကို ပြုလုပ်ပြီး ဇယားများ၊ ပျမ်းမျှရမှတ်များနှင့် မှတ်တမ်းများကို ကြည့်ပါ။",
        "About EduPredict AI": "EduPredict AI အကြောင်း", "A student performance prediction project": "ကျောင်းသားစွမ်းဆောင်ရည် ခန့်မှန်းရေး ပရောဂျက်",
        "Meet our team": "ကျွန်ုပ်တို့အဖွဲ့", "The people behind EduPredict AI": "EduPredict AI ကို ဖန်တီးသူများ",
        "Input categories": "ထည့်သွင်းရမည့် အမျိုးအစားများ", "Information used by the prediction model": "ခန့်မှန်းမော်ဒယ်အသုံးပြုသည့် အချက်အလက်များ",
        "Team Member": "အဖွဲ့ဝင်", "Language": "ဘာသာစကား", "English": "English",
        "Hours studied / day": "တစ်နေ့လျှင် စာလေ့လာချိန်", "Previous scores": "ယခင်ရမှတ်",
        "Attendance (%)": "တက်ရောက်မှု (%)", "Tutoring sessions": "သင်ကြားကူညီမှု အကြိမ်ရေ",
        "Motivation level": "လှုံ့ဆော်မှုအဆင့်", "Extracurricular activities": "သင်ရိုးပြင်ပ လှုပ်ရှားမှုများ",
        "Sleep hours / day": "တစ်နေ့လျှင် အိပ်ချိန်", "Physical activity (hours)": "ကိုယ်လက်လှုပ်ရှားမှု (နာရီ)",
        "Gender": "လိင်အမျိုးအစား", "Learning disabilities": "သင်ယူမှုအခက်အခဲ", "Internet access": "အင်တာနက်အသုံးပြုခွင့်",
        "Distance from home": "အိမ်မှအကွာအဝေး", "Teacher quality": "ဆရာ/ဆရာမ အရည်အသွေး",
        "School type": "ကျောင်းအမျိုးအစား", "Access to resources": "အရင်းအမြစ် အသုံးပြုခွင့်",
        "Parental involvement": "မိဘပါဝင်မှု", "Parental education": "မိဘပညာရေး",
        "Family income": "မိသားစုဝင်ငွေ", "Peer influence": "သူငယ်ချင်းများ၏ သက်ရောက်မှု",
        "GENERATE EXAM SCORE": "စာမေးပွဲရမှတ် ခန့်မှန်းရန်", "Download records as CSV": "မှတ်တမ်းများကို CSV အဖြစ် ဒေါင်းလုဒ်လုပ်ရန်",
        "Clear all prediction records": "ခန့်မှန်းမှုမှတ်တမ်းအားလုံး ဖျက်ရန်",
        "Prediction failed. Please check that the model and preprocessor match this app.": "ခန့်မှန်းမှု မအောင်မြင်ပါ။ မော်ဒယ်နှင့် preprocessor သည် ဤ app နှင့် ကိုက်ညီမှုရှိမရှိ စစ်ဆေးပါ။",
        "Excellent": "အလွန်ကောင်း", "Very Good": "အလွန်ကောင်းမွန်", "Good": "ကောင်း", "Average": "ပျမ်းမျှ", "Needs Improvement": "တိုးတက်ရန်လို",
        "Outstanding result. Keep the same learning routine and continue challenging yourself.": "အလွန်ကောင်းမွန်သော ရလဒ်ဖြစ်ပါသည်။ လက်ရှိသင်ယူမှုပုံစံကို ဆက်လက်ထိန်းသိမ်းပြီး ကိုယ့်ကိုယ်ကို စိန်ခေါ်ပါ။",
        "A strong result. Consistent revision and practice can help move performance even higher.": "ကောင်းမွန်သော ရလဒ်ဖြစ်ပါသည်။ ပုံမှန်ပြန်လည်လေ့လာခြင်းနှင့် လေ့ကျင့်ခြင်းဖြင့် ပိုမိုတိုးတက်နိုင်ပါသည်။",
        "Good progress. Focus on weaker topics and maintain a steady study schedule.": "တိုးတက်မှုကောင်းပါသည်။ အားနည်းသောဘာသာရပ်များကို အာရုံစိုက်ပြီး ပုံမှန်လေ့လာပါ။",
        "There is room to improve. Try regular practice, better attendance, and asking for support.": "တိုးတက်ရန်နေရာရှိပါသည်။ ပုံမှန်လေ့ကျင့်ခြင်း၊ တက်ရောက်မှုကောင်းမွန်ခြင်းနှင့် အကူအညီတောင်းခြင်းတို့ကို ပြုလုပ်ပါ။",
        "Start with small, achievable study goals. Review key lessons and build a consistent routine.": "အောင်မြင်နိုင်သော ရည်မှန်းချက်ငယ်များဖြင့် စတင်ပါ။ အရေးကြီးသင်ခန်းစာများကို ပြန်လည်လေ့လာပြီး ပုံမှန်အလေ့အကျင့်တည်ဆောက်ပါ။",
    }
}

def tr(text):
    return TRANSLATIONS.get(st.session_state.language, {}).get(text, text)

# -----------------------------
# Premium UI styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080d1b;
        --panel: #111a2e;
        --panel2: #15213a;
        --line: rgba(148,163,184,.16);
        --muted: #9aa9c4;
        --text: #edf3ff;
        --violet: #9b8cff;
        --cyan: #4de1df;
        --pink: #f28bc8;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(124,58,237,.22), transparent 34%),
          radial-gradient(ellipse at 95% 12%, rgba(6,182,212,.16), transparent 30%),
          linear-gradient(145deg, #080d1b 0%, #0b1325 48%, #0b1020 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(9, 15, 30, .96) !important; }
    [data-testid="stHeader"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button {
        color: #f4f7ff !important;
        background: linear-gradient(135deg, #26365a, #334a75) !important;
        border: 1px solid rgba(148,163,184,.38) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,.28) !important;
    }
    [data-testid="stHeader"] button svg,
    [data-testid="collapsedControl"] button svg,
    [data-testid="stSidebarCollapsedControl"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        stroke: #ffffff !important;
    }
    [data-testid="stHeader"] button:hover,
    [data-testid="collapsedControl"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
        background: linear-gradient(135deg, #7658e8, #168eaa) !important;
        border-color: rgba(77,225,223,.7) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13,21,40,.98), rgba(20,24,52,.98));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #e8edff; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2.6rem; max-width: 1500px; }
    h1,h2,h3,h4 { color: #f4f7ff; letter-spacing: -.035em; font-family: 'Manrope', sans-serif; }
    p, label, .stMarkdown { color: #d5def0; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    .topline {
        display:flex; align-items:center; justify-content:space-between; gap:12px;
        margin-bottom: 18px; color:#aebbd4; font-size:12px; letter-spacing:.12em;
        text-transform:uppercase; font-weight:800;
    }
    .hero {
        position:relative; overflow:hidden; padding:34px 38px; border-radius:26px;
        background: linear-gradient(115deg, rgba(91,63,181,.96), rgba(29,91,160,.92) 58%, rgba(12,145,158,.84));
        border:1px solid rgba(196,181,253,.35);
        box-shadow:0 22px 65px rgba(2,6,23,.35), inset 0 1px 0 rgba(255,255,255,.18);
        margin: 8px 0 26px;
    }
    .hero:before {
        content:""; position:absolute; width:340px; height:340px; right:-90px; top:-180px;
        border:1px solid rgba(255,255,255,.18); border-radius:50%;
        box-shadow:0 0 0 35px rgba(255,255,255,.035), 0 0 0 75px rgba(255,255,255,.025);
    }
    .hero-kicker { color:#d7e8ff; font-size:11px; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
    .hero h1 { color:white; font-size:clamp(30px,4vw,46px); font-weight:800; margin:8px 0 4px; }
    .hero p { color:#e2edff; font-size:15px; margin:0; max-width:660px; }
    .hero-chip {
        display:inline-block; margin-top:19px; padding:7px 12px; border-radius:999px;
        background:rgba(7,15,35,.22); border:1px solid rgba(255,255,255,.22);
        color:#effaff; font-size:11px; font-weight:700;
    }
    .section-head { display:flex; align-items:center; gap:10px; margin:24px 0 13px; }
    .section-icon {
        width:36px; height:36px; display:flex; align-items:center; justify-content:center;
        border-radius:12px; background:linear-gradient(135deg,rgba(155,140,255,.22),rgba(77,225,223,.12));
        border:1px solid rgba(155,140,255,.25); font-size:17px;
    }
    .section-title { font-family:'Manrope'; color:#f1f5ff; font-weight:800; font-size:18px; }
    .section-sub { color:#8fa0bd; font-size:12px; margin-top:1px; }
    .glass-card {
        background:linear-gradient(145deg,rgba(20,31,54,.92),rgba(15,24,43,.92));
        border:1px solid var(--line); border-radius:20px; padding:19px 20px;
        box-shadow:0 12px 32px rgba(0,0,0,.16);
    }
    .metric-card {
        background:linear-gradient(145deg,rgba(25,39,67,.96),rgba(16,26,47,.96));
        border:1px solid rgba(148,163,184,.16); border-radius:18px;
        padding:17px 19px; min-height:112px; box-shadow:0 10px 28px rgba(0,0,0,.14);
    }
    .metric-label { color:#9eacc5; font-size:11px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
    .metric-value { color:#f4f7ff; font-size:28px; font-weight:800; margin-top:8px; line-height:1.1; }
    .metric-note { color:#7f91b0; font-size:11px; margin-top:6px; }
    [data-testid="stForm"] {
        background:linear-gradient(145deg,rgba(17,27,48,.9),rgba(16,24,43,.88));
        border:1px solid rgba(148,163,184,.17); border-radius:22px; padding:20px;
        box-shadow:0 16px 40px rgba(0,0,0,.16);
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background:rgba(7,14,29,.68) !important; border:1px solid rgba(148,163,184,.22) !important;
        border-radius:12px !important; min-height:43px;
    }
    div[data-baseweb="input"] input, div[data-baseweb="select"] *, div[data-baseweb="textarea"] * {
        color:#eef4ff !important;
    }
    div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
        border-color:#59d9e5 !important; box-shadow:0 0 0 2px rgba(77,225,223,.13) !important;
    }
    [data-testid="stNumberInput"] button {
        width:36px !important; min-width:36px !important; height:36px !important;
        border-radius:50% !important; padding:0 !important; margin:0 3px !important;
        display:inline-flex !important; align-items:center !important; justify-content:center !important;
        transition:filter .15s ease, transform .15s ease, box-shadow .15s ease !important;
    }
    [data-testid="stNumberInput"] button:first-of-type {
        color:#ff5367 !important; background:rgba(239,68,68,.16) !important;
        border:1px solid rgba(248,113,113,.48) !important;
        box-shadow:0 3px 12px rgba(239,68,68,.10) !important;
    }
    [data-testid="stNumberInput"] button:last-of-type {
        color:#ffffff !important; background:#078bd0 !important;
        border:1px solid rgba(96,165,250,.65) !important;
        box-shadow:0 4px 14px rgba(14,165,233,.24) !important;
    }
    [data-testid="stNumberInput"] button:first-of-type svg { color:#ff5367 !important; stroke:#ff5367 !important; }
    [data-testid="stNumberInput"] button:last-of-type svg { color:#ffffff !important; stroke:#ffffff !important; }
    [data-testid="stNumberInput"] button:hover { filter:brightness(1.2); transform:scale(1.07); }
    .stButton button, .stFormSubmitButton button, .stDownloadButton button {
        border:0 !important; border-radius:12px !important; color:white !important;
        background:linear-gradient(100deg,#7658e8,#168eaa) !important;
        font-weight:800 !important; min-height:44px;
        box-shadow:0 9px 24px rgba(42,105,190,.24);
        transition:transform .18s ease, filter .18s ease;
    }
    .stButton button:hover, .stFormSubmitButton button:hover, .stDownloadButton button:hover {
        transform:translateY(-2px); filter:brightness(1.12);
    }
    .result-panel {
        background:linear-gradient(140deg,rgba(32,49,82,.96),rgba(21,34,61,.96));
        border:1px solid rgba(77,225,223,.25); border-radius:22px; padding:24px;
        box-shadow:0 16px 40px rgba(0,0,0,.2);
    }
    .score-number { font-family:'Manrope'; font-size:58px; line-height:1.05; font-weight:800; color:#58e1df; text-shadow:0 0 26px rgba(77,225,223,.2); }
    .score-caption { color:#9fb0ca; font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .level-pill { display:inline-block; padding:7px 12px; border-radius:999px; background:rgba(155,140,255,.16); border:1px solid rgba(155,140,255,.3); color:#d9d1ff; font-weight:800; font-size:12px; }
    .tip-box {
        background:linear-gradient(100deg,rgba(91,63,181,.24),rgba(8,145,178,.14));
        border:1px solid rgba(77,225,223,.22); border-left:4px solid #4de1df;
        padding:16px 18px; border-radius:15px; color:#dceaff; margin-top:14px;
    }
    .history-item {
        background:rgba(20,31,54,.8); border:1px solid rgba(148,163,184,.14);
        border-radius:15px; padding:13px 16px; margin:9px 0;
    }
    .history-date { color:#8193b2; font-size:11px; }
    .history-name { color:#f1f5ff; font-weight:800; }
    .history-score { color:#55dfdc; font-weight:800; }
    .sidebar-brand { padding:12px 4px 20px; }
    .sidebar-logo { font-size:27px; font-weight:800; color:white; font-family:'Manrope'; }
    .sidebar-caption { color:#91a2c1; font-size:11px; letter-spacing:.14em; text-transform:uppercase; }
    .sidebar-info { padding:14px; border:1px solid rgba(155,140,255,.2); border-radius:15px; background:rgba(91,63,181,.12); margin-top:12px; }
    .footer { text-align:center; color:#7385a4; font-size:11px; padding:22px 0 4px; }
    div[data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.18); border-radius:14px; overflow:hidden; }
    [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"] > div {
        background: linear-gradient(145deg, rgba(20,31,54,.96), rgba(15,24,43,.96)) !important;
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 18px;
        padding: 12px;
    }
    [data-testid="stVegaLiteChart"] canvas,
    [data-testid="stVegaLiteChart"] svg { border-radius: 12px; }
    [data-testid="stDataFrame"] iframe { border-radius: 14px; }
    .chart-card {
        background:linear-gradient(145deg,rgba(22,34,59,.98),rgba(12,22,41,.98));
        border:1px solid rgba(148,163,184,.17); border-radius:20px;
        padding:20px 20px 15px; min-height:310px;
        box-shadow:0 16px 36px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.035);
        overflow:hidden;
    }
    .chart-title-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:23px; }
    .chart-title { color:#f2f5ff; font-size:16px; font-weight:800; letter-spacing:-.02em; }
    .chart-subtitle { color:#8293b2; font-size:11px; margin-top:4px; }
    .chart-icon { width:36px; height:36px; display:flex; align-items:center; justify-content:center;
        border-radius:12px; color:#5de0e0; font-size:21px;
        background:rgba(77,225,223,.10); border:1px solid rgba(77,225,223,.18); }
    .trend-icon { color:#b3a7ff; background:rgba(155,140,255,.12); border-color:rgba(155,140,255,.2); }
    .bar-chart { display:flex; flex-direction:column; gap:17px; padding:4px 0 0; }
    .bar-row { display:grid; grid-template-columns:78px minmax(40px,1fr) 35px; gap:10px; align-items:center; }
    .bar-label { color:#aab8d1; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .bar-track { height:10px; border-radius:99px; background:rgba(129,148,180,.14); overflow:hidden; }
    .bar-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#7c6cf2,#45d6df);
        box-shadow:0 0 16px rgba(77,225,223,.22); min-width:2px; }
    .bar-value { color:#eff4ff; font-size:11px; font-weight:800; text-align:right; font-variant-numeric:tabular-nums; }
    .chart-axis { display:flex; justify-content:space-between; margin:12px 44px 0 88px; color:#7183a3; font-size:9px; }
    .line-chart-wrap { width:100%; margin-top:3px; }
    .line-chart-wrap svg { display:block; width:100%; height:auto; overflow:visible; }
    .chart-footnote { display:flex; align-items:center; gap:7px; color:#91a2c1; font-size:10px; margin-top:19px; }
    .chart-dot { width:7px; height:7px; border-radius:50%; background:#4de1df; box-shadow:0 0 9px rgba(77,225,223,.6); }
    .purple-dot { background:#9b8cff; box-shadow:0 0 9px rgba(155,140,255,.6); }
    .chart-unit { color:#6f82a4; }
    @media (max-width: 700px) {
        .chart-card { padding:16px 13px; min-height:285px; }
        .bar-row { grid-template-columns:62px minmax(30px,1fr) 30px; gap:7px; }
        .chart-axis { margin-left:69px; margin-right:37px; }
    }
    .stProgress > div > div > div > div { background:linear-gradient(90deg,#8b75ff,#4de1df); }
    hr { border-color:rgba(148,163,184,.14); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar navigation and details
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">✦ EduPredict</div>
            <div class="sidebar-caption">AI Student Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🌐 " + tr("Language"))
    st.radio("Language", ["English", "မြန်မာ"], key="language", horizontal=True, label_visibility="collapsed")
    st.markdown("### 🧭 " + tr("Workspace"))
    page_labels = {key: tr(key) for key in ["Predictor", "Analytics", "About Project"]}
    if "navigation_radio" not in st.session_state:
        st.session_state.navigation_radio = page_labels[st.session_state.page]
    if st.session_state.navigation_radio not in page_labels.values():
        st.session_state.navigation_radio = page_labels[st.session_state.page]
    selected_label = st.radio(
        tr("Navigate"), list(page_labels.values()),
        key="navigation_radio", label_visibility="collapsed",
    )
    page = next(key for key, label in page_labels.items() if label == selected_label)
    st.session_state.page = page
    st.divider()
    st.markdown("### ⚙️ " + tr("Model Information"))
    st.markdown(
        """
        <div class="sidebar-info">
            <div style="font-size:11px;color:#a99cff;font-weight:800;letter-spacing:.1em;">PREDICTION ENGINE</div>
            <div style="font-size:18px;color:#fff;font-weight:800;margin:5px 0;">Linear Regression</div>
            <div style="font-size:12px;color:#aab8d1;">Task · Regression</div>
            <div style="font-size:12px;color:#aab8d1;">Dataset · 6,607 records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("🔒 Student information is used for the current prediction session.")
    st.caption("Version 2.0 · Python + Machine Learning")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div class="topline">
        <span>✦ {tr("Intelligent learning insights")}</span>
        <span>{tr("STUDENT SUCCESS PLATFORM")}&nbsp; · &nbsp;2026</span>
    </div>
    <div class="hero">
        <div class="hero-kicker">✧ {tr("Machine learning dashboard")}</div>
        <h1>EduPredict AI</h1>
        <p>{tr("Understand student performance through data. Enter a learner profile and generate an estimated exam score with clear, actionable insights.")}</p>
        <span class="hero-chip">⚡ {tr("Smart prediction")} &nbsp; • &nbsp; 📊 {tr("Performance analytics")} &nbsp; • &nbsp; 🎓 {tr("Student-focused")}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

def section_heading(icon, title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-icon">{icon}</div>
            <div>
                <div class="section-title">{title}</div>
                <div class="section-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_card(label, value, note, icon):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="metric-label">{label}</div><div style="font-size:20px;">{icon}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Predictor page
# -----------------------------
if page == "Predictor":
    total = len(st.session_state.history)
    average = (
        sum(x["Predicted Score"] for x in st.session_state.history) / total
        if total else 0
    )
    best = max((x["Predicted Score"] for x in st.session_state.history), default=0)

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card(tr("Total predictions"), total, tr("Records in this session"), "🧾")
    with m2:
        metric_card(tr("Average score"), f"{average:.1f}" if total else "—", tr("Across all predictions"), "📈")
    with m3:
        metric_card(tr("Highest score"), f"{best:.1f}" if total else "—", tr("Best predicted result"), "🏆")

    section_heading("🪄", tr("Build a student profile"), tr("Complete the details below to create a prediction"))
    st.markdown(
        '<div class="glass-card" style="margin-bottom:14px;">'
        '<span style="color:#a99cff;font-weight:800;font-size:11px;letter-spacing:.12em;">PROFILE SETUP</span>'
        '<span style="color:#93a4c1;font-size:12px;"> &nbsp;·&nbsp; 19 input features &nbsp;·&nbsp; Choose every option; student name is optional</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("student_form", clear_on_submit=False):
        student_name = st.text_input(
            "👤 " + tr("Student name (optional)"),
            placeholder="e.g. Alex Morgan",
            help="Leave blank to use an automatically generated student label.",
        )
        tab_academic, tab_lifestyle, tab_environment = st.tabs(
            ["📚 Academic", "🌿 Lifestyle", "🏫 Learning environment"]
        )

        with tab_academic:
            st.markdown("#### Academic performance")
            a1, a2, a3 = st.columns(3)
            with a1:
                hours_studied = st.number_input("⏱️ " + tr("Hours studied / day"), min_value=0, max_value=24, value=0, step=1)
                previous_scores = st.number_input("🎯 " + tr("Previous scores"), min_value=0, max_value=100, value=0, step=1)
            with a2:
                attendance = st.number_input("📅 " + tr("Attendance (%)"), min_value=0, max_value=100, value=0, step=1)
                tutoring_sessions = st.number_input("🧑‍🏫 " + tr("Tutoring sessions"), min_value=0, max_value=20, value=0, step=1)
            with a3:
                motivation_level = st.selectbox("🔥 " + tr("Motivation level"), ["— Select an option —", "Low", "Medium", "High"], index=0)
                extracurricular_activities = st.selectbox("🎨 " + tr("Extracurricular activities"), ["— Select an option —", "Yes", "No"], index=0)

        with tab_lifestyle:
            st.markdown("#### Health, routine & personal profile")
            l1, l2, l3 = st.columns(3)
            with l1:
                sleep_hours = st.number_input("😴 " + tr("Sleep hours / day"), min_value=0, max_value=24, value=0, step=1)
                physical_activity = st.number_input("🏃 " + tr("Physical activity (hours)"), min_value=0, max_value=24, value=0, step=1)
            with l2:
                gender = st.selectbox("🧑 " + tr("Gender"), ["— Select an option —", "Male", "Female"], index=0)
                learning_disabilities = st.selectbox("🧩 " + tr("Learning disabilities"), ["— Select an option —", "No", "Yes"], index=0)
            with l3:
                internet_access = st.selectbox("🌐 " + tr("Internet access"), ["— Select an option —", "Yes", "No"], index=0)
                distance_from_home = st.selectbox("📍 " + tr("Distance from home"), ["— Select an option —", "Near", "Moderate", "Far"], index=0)

        with tab_environment:
            st.markdown("#### School, family & resources")
            e1, e2, e3 = st.columns(3)
            with e1:
                teacher_quality = st.selectbox("⭐ " + tr("Teacher quality"), ["— Select an option —", "Low", "Medium", "High"], index=0)
                school_type = st.selectbox("🏛️ " + tr("School type"), ["— Select an option —", "Public", "Private"], index=0)
                access_to_resources = st.selectbox("📖 " + tr("Access to resources"), ["— Select an option —", "Low", "Medium", "High"], index=0)
            with e2:
                parental_involvement = st.selectbox("👨‍👩‍👧 " + tr("Parental involvement"), ["— Select an option —", "Low", "Medium", "High"], index=0)
                parental_education_level = st.selectbox("🎓 " + tr("Parental education"), ["— Select an option —", "High School", "College", "Postgraduate"], index=0)
                family_income = st.selectbox("💼 " + tr("Family income"), ["— Select an option —", "Low", "Medium", "High"], index=0)
            with e3:
                peer_influence = st.selectbox("🤝 " + tr("Peer influence"), ["— Select an option —", "Positive", "Neutral", "Negative"], index=0)

        st.markdown("---")
        submitted = st.form_submit_button("🚀  " + tr("GENERATE EXAM SCORE"), use_container_width=True, type="primary")

    if submitted:
        required_choices = {
            "Motivation level": motivation_level,
            "Extracurricular activities": extracurricular_activities,
            "Gender": gender,
            "Learning disabilities": learning_disabilities,
            "Internet access": internet_access,
            "Distance from home": distance_from_home,
            "Teacher quality": teacher_quality,
            "School type": school_type,
            "Access to resources": access_to_resources,
            "Parental involvement": parental_involvement,
            "Parental education": parental_education_level,
            "Family income": family_income,
            "Peer influence": peer_influence,
        }
        missing_choices = [
            field for field, value in required_choices.items()
            if value == "— Select an option —"
        ]
        if missing_choices:
            st.error(
                "⚠️ Please select all required options before generating a score. "
                "The following fields are still empty:"
            )
            missing_fields = missing_choices
            st.markdown(
                '<div style="background:rgba(248,113,113,.10);border:1px solid rgba(248,113,113,.35);'
                'border-left:4px solid #f87171;border-radius:13px;padding:13px 16px;color:#ffd6d6;">'
                + "<br>".join("• " + field for field in missing_fields)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.stop()

        input_data = pd.DataFrame([{
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Parental_Involvement": parental_involvement,
            "Access_to_Resources": access_to_resources,
            "Extracurricular_Activities": extracurricular_activities,
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Motivation_Level": motivation_level,
            "Internet_Access": internet_access,
            "Tutoring_Sessions": tutoring_sessions,
            "Family_Income": family_income,
            "Teacher_Quality": teacher_quality,
            "School_Type": school_type,
            "Peer_Influence": peer_influence,
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": learning_disabilities,
            "Parental_Education_Level": parental_education_level,
            "Distance_from_Home": distance_from_home,
            "Gender": gender,
        }])

        try:
            with st.spinner("🧠 EduPredict AI is analyzing the student profile and generating the exam score..."):
                processed_data = preprocessor.transform(input_data)
                prediction = model.predict(processed_data)[0]
                score = max(0, min(100, float(prediction)))
        except Exception as exc:
            st.error(tr("Prediction failed. Please check that the model and preprocessor match this app."))
            st.exception(exc)
            st.stop()

        if score >= 90:
            level = "Excellent"
            emoji = "🌟"
            advice = "Outstanding result. Keep the same learning routine and continue challenging yourself."
        elif score >= 80:
            level = "Very Good"
            emoji = "🎉"
            advice = "A strong result. Consistent revision and practice can help move performance even higher."
        elif score >= 70:
            level = "Good"
            emoji = "👍"
            advice = "Good progress. Focus on weaker topics and maintain a steady study schedule."
        elif score >= 60:
            level = "Average"
            emoji = "📘"
            advice = "There is room to improve. Try regular practice, better attendance, and asking for support."
        else:
            level = "Needs Improvement"
            emoji = "💪"
            advice = "Start with small, achievable study goals. Review key lessons and build a consistent routine."

        record = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Student": student_name.strip() or f"Student {len(st.session_state.history) + 1}",
            "Predicted Score": round(score, 2),
            "Performance": level,
        }
        st.session_state.history.append(record)
        st.session_state.latest_result = record
        # Re-run once so Predictor KPI cards update immediately after submission.
        st.rerun()

    if "latest_result" in st.session_state:
        result = st.session_state.latest_result
        score = result["Predicted Score"]
        level = result["Performance"]
        emoji = {"Excellent":"🌟","Very Good":"🎉","Good":"👍","Average":"📘","Needs Improvement":"💪"}.get(level, "🎓")
        advice = {
            "Excellent":"Outstanding result. Keep the same learning routine and continue challenging yourself.",
            "Very Good":"A strong result. Consistent revision and practice can help move performance even higher.",
            "Good":"Good progress. Focus on weaker topics and maintain a steady study schedule.",
            "Average":"There is room to improve. Try regular practice, better attendance, and asking for support.",
            "Needs Improvement":"Start with small, achievable study goals. Review key lessons and build a consistent routine.",
        }.get(level, "")

        section_heading("🎯", tr("Prediction result"), tr("Your latest student performance estimate"))
        r1, r2 = st.columns([1.05, 1])
        with r1:
            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="score-caption">{tr("Predicted exam score")}</div>
                    <div style="display:flex;align-items:baseline;gap:8px;margin:12px 0 8px;">
                        <div class="score-number">{score:.2f}</div>
                        <div style="color:#9aacc8;font-size:15px;font-weight:700;">/ 100</div>
                    </div>
                    <div class="level-pill">{emoji} &nbsp; {tr(level)}</div>
                    <div style="margin-top:20px;color:#9aacc8;font-size:12px;">{tr("Performance indicator")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(score / 100)
            st.caption(f"{score:.1f}% predicted score")
        with r2:
            st.markdown(
                f"""
                <div class="result-panel" style="height:100%;">
                    <div class="score-caption">{tr("Learner profile")}</div>
                    <div style="font-size:25px;font-weight:800;color:#f4f7ff;margin:12px 0 4px;">{result['Student']}</div>
                    <div style="color:#9aacc8;font-size:12px;">Generated {result['Date']}</div>
                    <div style="border-top:1px solid rgba(148,163,184,.16);margin:18px 0;"></div>
                    <div style="font-size:13px;color:#dce8ff;line-height:1.7;">{tr(advice)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption("Note: This is a machine-learning estimate, not a guaranteed exam result.")

# -----------------------------
# Analytics page
# -----------------------------
elif page == "Analytics":
    section_heading("📊", tr("Performance analytics"), tr("Review predictions generated in this session"))
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        total = len(history_df)
        avg = history_df["Predicted Score"].mean()
        high = history_df["Predicted Score"].max()
        low = history_df["Predicted Score"].min()
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Predictions", total, "Total saved records", "🧾")
        with c2: metric_card(tr("Average"), f"{avg:.2f}", "Mean predicted score", "📈")
        with c3: metric_card("Highest", f"{high:.2f}", "Top predicted score", "🏆")
        with c4: metric_card("Lowest", f"{low:.2f}", "Lowest predicted score", "📉")

        section_heading("📉", "Score distribution", "Compare each student's predicted result")
        chart_col1, chart_col2 = st.columns(2)

        # Custom charts use HTML/CSS/SVG; no Plotly dependency is required.
        chart_rows = history_df.tail(8).reset_index(drop=True)
        chart_labels = [html.escape(str(x)[:15]) for x in chart_rows["Student"]]
        chart_scores = [float(x) for x in chart_rows["Predicted Score"]]

        with chart_col1:
            bars_html = ""
            for label, value in zip(chart_labels, chart_scores):
                bars_html += f"""
                <div class="bar-row">
                    <div class="bar-label" title="{label}">{label}</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{max(0, min(100, value)):.2f}%"></div></div>
                    <div class="bar-value">{value:.1f}</div>
                </div>
                """
            st.markdown(
                f"""
                <div class="chart-card">
                    <div class="chart-title-row">
                        <div><div class="chart-title">Predicted scores</div>
                        <div class="chart-subtitle">Latest {len(chart_scores)} student results</div></div>
                        <div class="chart-icon">▥</div>
                    </div>
                    <div class="bar-chart">{bars_html}</div>
                    <div class="chart-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
                    <div class="chart-footnote"><span class="chart-dot"></span> Predicted exam score <span class="chart-unit">/ 100</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chart_col2:
            width, height = 520, 220
            left, right, top, bottom = 34, 16, 18, 34
            plot_w, plot_h = width - left - right, height - top - bottom
            points = []
            for i, value in enumerate(chart_scores):
                x = left + (plot_w * i / max(1, len(chart_scores) - 1))
                y = top + (100 - max(0, min(100, value))) / 100 * plot_h
                points.append((x, y))
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            circles = "".join(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#9b8cff" stroke="#eef0ff" stroke-width="2"/>'
                for x, y in points
            )
            area = f"{left},{height-bottom} " + polyline + f" {left + plot_w},{height-bottom}"
            grid = "".join(
                f'<g><line x1="{left}" y1="{top + plot_h*j/4:.1f}" x2="{width-right}" y2="{top + plot_h*j/4:.1f}" stroke="rgba(169,184,211,.16)" stroke-dasharray="4 6"/><text x="4" y="{top + plot_h*j/4 + 4:.1f}" fill="#8293b2" font-size="10">{100-j*25}</text></g>'
                for j in range(5)
            )
            xlabels = "".join(
                f'<text x="{x:.1f}" y="{height-9}" text-anchor="middle" fill="#8293b2" font-size="10">{i+1}</text>'
                for i, (x, _) in enumerate(points)
            )
            st.markdown(
                f"""
                <div class="chart-card">
                    <div class="chart-title-row">
                        <div><div class="chart-title">Prediction trend</div>
                        <div class="chart-subtitle">Score movement across predictions</div></div>
                        <div class="chart-icon trend-icon">⌁</div>
                    </div>
                    <div class="line-chart-wrap">
                    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Prediction trend line chart">
                        <defs><linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#9b8cff"/><stop offset="100%" stop-color="#4de1df" stop-opacity="0"/></linearGradient></defs>
                        {grid}
                        <polygon points="{area}" fill="url(#trendFill)" opacity=".26"/>
                        <polyline points="{polyline}" fill="none" stroke="#9b8cff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
                        {circles}{xlabels}
                    </svg>
                    </div>
                    <div class="chart-footnote"><span class="chart-dot purple-dot"></span> Prediction sequence <span class="chart-unit">· latest first to last</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        section_heading("🗂️", "Prediction records", "A polished overview of every student prediction")
        # Stylish record cards make the table easier to scan on smaller screens.
        for idx, item in history_df.iloc[::-1].iterrows():
            score_value = float(item["Predicted Score"])
            level_color = "#4de1df" if score_value >= 80 else "#9b8cff" if score_value >= 60 else "#f2a6a0"
            st.markdown(
                f"""
                <div class="history-item" style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;">
                    <div style="min-width:180px;">
                        <div class="history-date">#{idx + 1} &nbsp;·&nbsp; {item['Date']}</div>
                        <div class="history-name" style="font-size:15px;margin-top:5px;">👤 {item['Student']}</div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="padding:6px 11px;border-radius:999px;background:rgba(155,140,255,.13);border:1px solid rgba(155,140,255,.25);color:#d8d0ff;font-size:11px;font-weight:800;">{tr(item['Performance'])}</span>
                        <span style="font-size:20px;font-weight:800;color:{level_color};">{score_value:.2f}<span style="font-size:11px;color:#91a2c1;"> / 100</span></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📋 View data table", expanded=False):
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )
        csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️  " + tr("Download records as CSV"),
            data=csv_data,
            file_name="edupredict_student_records.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("🗑️ " + tr("Clear all prediction records"), type="secondary"):
            st.session_state.history = []
            st.session_state.pop("latest_result", None)
            st.rerun()
    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;padding:42px 20px;">
                <div style="font-size:42px;">📊</div>
                <div style="font-size:20px;font-weight:800;color:#f4f7ff;margin:10px 0;">No analytics yet</div>
                <div style="color:#91a2c1;font-size:13px;">Create your first student prediction to see charts, averages, and records here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------
# About page
# -----------------------------
else:
    section_heading("🎓", tr("About EduPredict AI"), tr("A student performance prediction project"))
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-size:22px;font-weight:800;color:#f4f7ff;">Making student data easier to understand.</div>
            <p style="color:#aab8d1;line-height:1.8;margin-top:10px;">
                EduPredict AI is a machine-learning application that estimates a student's exam score
                from academic, personal, family, and school-related information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    about1, about2, about3 = st.columns(3)
    with about1:
        st.markdown('<div class="glass-card"><div style="font-size:25px;">🧠</div><h3>Machine Learning</h3><p style="color:#9aabc6;font-size:13px;">Uses a trained Linear Regression model to estimate exam scores.</p></div>', unsafe_allow_html=True)
    with about2:
        st.markdown('<div class="glass-card"><div style="font-size:25px;">📝</div><h3>Student profile</h3><p style="color:#9aabc6;font-size:13px;">Uses 19 input features covering study habits and learning environment.</p></div>', unsafe_allow_html=True)
    with about3:
        st.markdown('<div class="glass-card"><div style="font-size:25px;">📊</div><h3>Analytics</h3><p style="color:#9aabc6;font-size:13px;">Stores session predictions and displays charts with CSV export.</p></div>', unsafe_allow_html=True)

    section_heading("👥", tr("Meet our team"), tr("The people behind EduPredict AI"))
    st.markdown(
        """
        <div style="background:linear-gradient(120deg,rgba(91,63,181,.20),rgba(22,145,170,.14));
                    border:1px solid rgba(155,140,255,.25);border-radius:18px;
                    padding:17px 20px;margin-bottom:14px;">
            <div style="color:#a99cff;font-size:10px;font-weight:800;letter-spacing:.16em;">PROJECT TEAM</div>
            <div style="font-size:20px;font-weight:800;color:#f4f7ff;margin-top:5px;">PHOENIX</div>
            <div style="font-size:12px;color:#9aabc6;margin-top:3px;">Student Project Members</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    team_members = [
        ("442", "Tun", "🧑‍💻", "Team Member"),
        ("563", "Dia", "👩‍💻", "Team Member"),
        ("509", "VexX", "🧑‍🎨", "Team Member"),
        ("461", "Rika", "👩‍🎓", "Team Member"),
        ("573", "Xiao Yan", "🧑‍🚀", "Team Member"),
        ("441", "Gavi", "🧑‍🚀", "Team Member"),
        ("313", "Bright", "🧑‍💻", "Team Member"),
        ("583", "Tulip", "🌷", "Team Member"),
    ]
    team_cols = st.columns(3)
    for member_index, (number, member_name, member_icon, role) in enumerate(team_members):
        with team_cols[member_index % 3]:
            st.markdown(
                f"""
                <div style="background:linear-gradient(145deg,rgba(24,38,66,.96),rgba(15,25,46,.96));
                            border:1px solid rgba(148,163,184,.18);border-radius:18px;
                            padding:20px 17px;margin-bottom:13px;min-height:142px;
                            box-shadow:0 12px 28px rgba(0,0,0,.15);">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="width:42px;height:42px;display:flex;align-items:center;justify-content:center;
                                    border-radius:14px;background:rgba(155,140,255,.15);
                                    border:1px solid rgba(155,140,255,.25);font-size:22px;">{member_icon}</div>
                        <span style="font-size:9px;color:#8fa0bd;letter-spacing:.1em;font-weight:800;">{tr(role).upper()}</span>
                    </div>
                    <div style="font-size:18px;font-weight:800;color:#f4f7ff;margin-top:13px;">{member_name}</div>
                    <div style="font-size:11px;color:#65cfe2;margin-top:4px;letter-spacing:.06em;">NUCS-2410{number}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_heading("🧩", tr("Input categories"), tr("Information used by the prediction model"))
    st.markdown(
        """
        - **Academic:** study hours, attendance, previous scores, tutoring, motivation
        - **Lifestyle:** sleep, physical activity, gender, internet access
        - **Learning environment:** teacher quality, school type, learning resources, parental involvement, parental education, family income, peer influence, extracurricular activities, learning disabilities, distance from home
        """
    )
    st.info("The prediction depends on the quality and compatibility of the trained model and its preprocessor. The result should be treated as an estimate, not a final judgment about a student.")

st.markdown(
    """
    <div class="footer">
        ✦ <b>EduPredict AI</b> &nbsp;·&nbsp; Student Performance Prediction System<br>
        Built with Python, Streamlit & Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)
