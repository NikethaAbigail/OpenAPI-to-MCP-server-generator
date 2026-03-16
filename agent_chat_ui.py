import streamlit as st
import os, re, json, time, yaml, asyncio
from pathlib import Path
from huggingface_hub import InferenceClient

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & UI STYLING
# ──────────────────────────────────────────────────────────────────────────────
HF_TOKEN = "enter the token" 
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

st.set_page_config(page_title="Llama AI Agent", page_icon="🤖", layout="wide")

# CUSTOM CSS FOR CREATIVE UI AND LARGER TEXT
st.markdown("""
    <style>
    /* Global Font Size Increase */
    html, body, [class*="css"]  {
        font-size: 18px !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 42px !important;
        font-weight: 800;
        background: -webkit-linear-gradient(#00dbde, #fc00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 20, 30, 0.8);
        border-right: 1px solid #444;
    }
    
    /* Chat Bubble Enhancements */
    [data-testid="stChatMessage"] {
        background-color: #1e1e2f !important;
        border-radius: 15px !important;
        border: 1px solid #3d3d5c !important;
        margin-bottom: 10px;
    }
    
    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #6200ea, #03dac6);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(3, 218, 198, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "spec" not in st.session_state:
    st.session_state.spec = None

client = InferenceClient(api_key=HF_TOKEN)

# ──────────────────────────────────────────────────────────────────────────────
# 2. CORE AGENT LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def get_tool_prompt(spec):
    paths = list(spec.get("paths", {}).items())[:5] 
    tool_desc = []
    for path, path_item in paths:
        for method, op in path_item.items():
            if method.upper() in ["GET", "POST"]:
                name = op.get("operationId") or f"{method}_{path}"
                tool_desc.append(f"🛠️ **{name}**: {op.get('summary', 'No description')}")
    return "\n".join(tool_desc)

async def run_agent_turn(prompt, spec):
    history = st.session_state.messages[-2:] if len(st.session_state.messages) > 2 else []
    tools_list = get_tool_prompt(spec)
    system_msg = (
        f"You are a Creative API Agent. Use these available tools:\n{tools_list}\n"
        "Be helpful, use emojis, and explain which tool you are using."
    )
    messages = [{"role": "system", "content": system_msg}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(model=MODEL_ID, messages=messages, max_tokens=600)
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ **Error**: {str(e)}"

# ──────────────────────────────────────────────────────────────────────────────
# 3. INTERACTIVE UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 OpenAPI-to-MCP Interactive Agent</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    spec_file = st.file_uploader("📥 Upload OpenAPI Spec", type=["yaml", "yml", "json"])
    
    if st.button("🚀 Initialize Agent") and spec_file:
        with st.status("Parsing Specification...", expanded=True) as status:
            try:
                raw = spec_file.read().decode("utf-8")
                st.session_state.spec = yaml.safe_load(raw)
                st.success("Tools Integrated Successfully!")
                status.update(label="Agent Ready!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Failed: {e}")

# Display chat history with larger fonts
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div style="font-size: 20px;">{msg["content"]}</div>', unsafe_allow_html=True)

# User Input
if prompt := st.chat_input("Ask your AI Agent anything..."):
    if not st.session_state.spec:
        st.warning("⚠️ Please upload and initialize your API spec first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div style="font-size: 20px;">{prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            with st.spinner("🔮 Agent is analyzing tools..."):
                reply = asyncio.run(run_agent_turn(prompt, st.session_state.spec))
                st.markdown(f'<div style="font-size: 20px;">{reply}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": reply})