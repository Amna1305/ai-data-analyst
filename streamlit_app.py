import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom Header Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    h1 {
        text-align: center;
        color: #2E86C1;
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        margin-top: -15px;
        color: #5D6D7E;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #AAB7B8;
        margin-top: 50px;
    }
    </style>

    <div class="main">
        <h1>📊 AI-Powered Data Analyst</h1>
        <p class="subheader">Upload your CSV file and ask questions like you're talking to a real analyst.</p>
    </div>
""", unsafe_allow_html=True)

# --- File Upload Section ---
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# --- Sample CSV Option ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🧪 Use Sample Data"):
        uploaded_file = "employee_data.csv"
        st.success("Sample employee data loaded.")

# --- Load & Analyze ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        question = st.text_input("🔍 Ask a question about your data:")

        if question:
            # --- Setup Groq LLM ---
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama3-8b-8192",
                temperature=0
            )

            # --- Create Agent ---
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                allow_dangerous_code=True
            )

            with st.spinner("🧠 Thinking..."):
                try:
                    response = agent.run(question)
                    st.success("✅ Answer Ready")
                    st.markdown(f"""
                        <div style='background-color:#f4f6f7; padding:15px; border-left:5px solid #2ECC71; border-radius:6px;'>
                            <b>Answer:</b> {response}
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Agent Error: {e}")

    except Exception as e:
        st.error(f"❌ File Read Error: {str(e)}")

# --- Footer ---
st.markdown("""
    <hr>
    <div class="footer">
        Built by <b>Amna Faisal</b> — Freelance AI Developer 👩‍💻 | Available on Fiverr & Upwork
    </div>
""", unsafe_allow_html=True)
