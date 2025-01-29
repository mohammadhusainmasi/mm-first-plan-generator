from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import Together
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st
import time

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Please check your .env file.")
    st.stop()
if not TOGETHER_API_KEY:
    st.warning("Missing TOGETHER_API_KEY. Only Gemini will be available.")

# Categories
categories = {
    "Programming 💻": 0.1,
    "Art 🎨": 0.2,
    "Music 🎶": 0.3,
    "Fitness 🏍️": 0.4,
    "Cooking 🍳": 0.5,
    "Languages 🌍": 0.6,
    "Learning ✍️": 0.7,
    "Business 📈": 0.8,
}

# Languages
available_languages = [
   
 "English 📚", "Spanish 📘", "French 📙", "German 📔", "Hindi 📗", "Chinese 📚", "Japanese 📖", "Gujarati 📕"
]
# Prompt Template
plan_template = PromptTemplate(
    template="""
🎯 **Your Task**:  
You are a **senior {category} professional**, ranked in the **top 1%**. Generate a **personalized plan** to master **{skill}** in **{category}**.  
🗓 **Details**:  
1️⃣ **Skill**: {skill}  
2️⃣ **Category**: {category}  
3️⃣ **Days**: {days_available}  
4️⃣ **Daily Time**: {daily_time} hours  
5️⃣ **Language**: {language}  
📌 **Now, create a structured learning plan** in **{language}**.
""",
    input_variables=["skill", "category", "days_available", "daily_time", "language"]
)

# Streamlit UI
st.title("🎯 Personal Plan Generator")
st.subheader("🚀 Generate a step-by-step plan to master a skill using AI")

# UI Inputs
selected_category = st.selectbox("Select a Category", options=categories.keys())
model_choice = st.selectbox(
    "Select AI Model",
    ["Gemini (Google)", "Together AI", "Meta Llama 3.3 70B Instruct Turbo"]
)
skill = st.text_input("Enter the skill you want to master", placeholder="E.g., Python programming")
days_available = st.number_input("Number of days available", min_value=1, max_value=365, value=10)
daily_time = st.number_input("Daily time commitment (hours)", min_value=1, max_value=24, value=2)
selected_language = st.selectbox("Select Output Language", available_languages)
temperature = st.slider(" 🌡️ Select Model Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Initialize Selected AI Model
ai_model = None
try:
    if model_choice == "Gemini (Google)":
        ai_model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature
        )
    elif model_choice == "Together AI":
        ai_model = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            together_api_key=TOGETHER_API_KEY
        )
    elif model_choice == "Meta Llama 3.3 70B Instruct Turbo":
        try:
            ai_model = Together(
                model="meta-llama/Meta-Llama-3-70B-Instruct",
                together_api_key=TOGETHER_API_KEY
            )
        except Exception as e:
            st.warning(f"⚠️ Meta Llama 3.3 70B model is unavailable. Error: {str(e)}. Falling back to Mistral 7B.")
            ai_model = Together(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                together_api_key=TOGETHER_API_KEY
            )
except Exception as e:
    st.error(f"❌ Error initializing model: {e}")
    st.stop()

# Retry logic for generating the plan with a fallback
def generate_plan():
    plan_chain = LLMChain(prompt=plan_template, llm=ai_model)
    
    retries = 3
    for attempt in range(retries):
        try:
            response = plan_chain.run({
                "skill": skill,
                "category": selected_category,
                "days_available": days_available,
                "daily_time": daily_time,
                "language": selected_language,
            })
            return response.strip() if isinstance(response, str) else str(response).strip()
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed. Retrying... ({str(e)})")
            time.sleep(2)  # Wait before retrying

    # If all retries fail
    st.error(f"❌ Error generating plan after {retries} attempts: {e}")
    st.exception(e)  # Show stack trace for debugging
    return None

# Generate Button
if st.button("Generate Plan"):
    if not skill:
        st.error("Please enter a skill to master.")
    else:
        with st.spinner(f"📝 Generating plan using {model_choice} in {selected_language}..."):
            plan = generate_plan()
            if plan:
                st.markdown(f"## 🌍 Plan in {selected_language}:")
                st.markdown(plan)
            else:
                st.error("⚠️ Unable to generate a plan after multiple attempts. Please try again.")
