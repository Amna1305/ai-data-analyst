# 📊 AI Data Analyst – CSV Insight Generator

A smart, AI-powered tool that transforms any CSV file into real-time business insights. Built using **LangChain**, **Groq**, **Pandas**, and **Streamlit**.

> Upload your CSV, ask in plain English – get instant, code-backed answers.

---

## 🚀 Key Features

✅ Upload any CSV file  
✅ Ask natural language questions like:  
  • “What’s the average salary by department?”  
  • “Who are the top 5 highest-paid employees?”  
✅ Real-time Python code execution  
✅ Uses Groq’s blazing-fast LLMs  
✅ Ready for client projects (Fiverr/Upwork)

---

## 🛠️ How to Use Locally

1. Clone this repo 
  
   git clone https://github.com/Amna1305/ai-data-analyst.git
   cd ai-data-analyst


2. Set up your .env file
Create a file named .env and add:

GROQ_API_KEY=your_actual_api_key_here

3. Install dependencies

pip install -r requirements.txt

4. Run the app


streamlit run streamlit_app.py

## 🎯Use Cases
Quick insights for business reports

Data summaries for non-technical teams

Department-level breakdowns

Ideal for Fiverr/Upwork data analysis gigs

## 📁Project Structure
bash
Copy
Edit
data_analyst_agent/
├── streamlit_app.py       # Main Streamlit app
├── app.py                 # Agent logic (optional CLI use)
├── requirements.txt       # All Python packages
├── .env.example           # API key template
├── employee_data.csv      # Sample dataset
└── README.md              # This file

## 🧠 Built With
LangChain

Groq API

Streamlit

Pandas

Python 3.11

## ✨ Author
Amna Faisal
Freelance AI Developer | Computer Science Student
🔗 LinkedIn
💼 Portfolio: Coming soon
📫 Available on Fiverr & Upwork

📎 License
This project is open-source and ready to customize for client or commercial use.