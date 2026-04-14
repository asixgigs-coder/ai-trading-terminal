import streamlit as st
import json
import os
import subprocess

STATUS_FILE = "status.json"
LOG_FILE = "signals.json"

st.set_page_config(layout="wide")
st.title("📱 AI Trading Control Panel")

def get_status():
    if not os.path.exists(STATUS_FILE):
        return False
    with open(STATUS_FILE, "r") as f:
        return json.load(f).get("running", False)

status = get_status()
st.metric("Bot Status", "🟢 Running" if status else "🔴 Stopped")

col1, col2 = st.columns(2)

if col1.button("▶️ Start Bot"):
    subprocess.Popen(["python", "bot.py"])
    st.success("Bot started")

if col2.button("⏹ Stop Bot"):
    os.system("pkill -f bot.py")
    st.warning("Bot stopped")

st.markdown("## 📊 Recent Signals")

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
        for log in logs[:10]:
            st.write(log)
else:
    st.info("No signals yet")

if st.button("🔎 Run Scan Now"):
    os.system("python bot.py")
    st.success("Scan executed")
