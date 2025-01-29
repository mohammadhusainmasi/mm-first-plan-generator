from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st
import time
import tenacity
import google.api_core.exceptions

# Load environment variables
load_dotenv()

# API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Please check your .env file.")
    st.stop()

# Categories for selection
categories = {
    "Programming 💻": 0.1,
    "Art 🎨": 0.2,
    "Music 🎶": 0.3,
    "Fitness 🏋️": 0.4,
    "Cooking 🍳": 0.5,
    "Languages 🌍": 0.6,
    "Learning ✍️": 0.7,
    "Business 📈": 0.8,
}

# Available languages
available_languages = [
    "English", "Spanish", "French", "German", 
    "Hindi", "Chinese", "Japanese", "Gujarati"
]

# Personal Plan Template
plan_template = """
🎯 **Your Task**:  
You are a **senior {category} professional**, renowned for your expertise and ranked in the **top 1%** of the market. Using your vast knowledge and experience, generate a **personalized plan** to master **{skill}** in the category of **{category}**, creating a significant impact by ensuring daily measurable progress.  

📅 **Plan Details**:  
1️⃣ **Skill to Master**: {skill}  
2️⃣ **Category**: {category}  
3️⃣ **Total Days Available**: {days_available}  
4️⃣ **Daily Time Commitment**: {daily_time} hours  
5️⃣ **Language**: {language}

📝 **Plan Requirements**:  
✅ Each day MUST include a **clear, actionable objective** with subtopics to master. No days should be skipped or merged into intervals.  
✅ Allocate **specific activities** for every single day, ensuring equal focus on **learning**, **practicing**, and **reviewing**.  
✅ Include milestones as **additional tasks** (not skipping days) for motivation and tracking.  
✅ Focus on practical learning with examples and exercises to solidify knowledge.  
✅ Use **emojis** and engaging language to make the plan approachable and fun.  

📌 **Now, create a step-by-step plan** in **{language}** to help achieve mastery in **{skill}** within **{days_available}** days, dedicating **{daily_time}** hours daily.
"""

# Streamlit UI
st.title("🎯 Personal Plan Generator")
st.subheader(" 🚀 Generate a step-by-step plan to master a skill using Generative AI")

# Category selection
selected_category = st.selectbox(
    "Select a Category",
    options=categories.keys(),
    help="Choose a category related to the skill you want to master."
)

# Skill Input
skill = st.text_input("Enter the skill you want to master", placeholder="E.g., Python programming")

# Number of days
days_available = st.number_input("Number of days available", min_value=1, max_value=365, value=10)

# Daily time commitment
daily_time = st.number_input("Daily time commitment (hours)", min_value=1, max_value=24, value=2)

# **Language Selection (Dropdown)**
selected_language = st.selectbox("Select Output Language", available_languages)

# **Manual Temperature Selection**
temperature = st.slider(
    "Select Model Creativity (Temperature)", 
    min_value=0.0, max_value=1.0, 
    value=0.7, step=0.1
)

# Initialize Gemini Model with manually set temperature
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=temperature
)

# Function to generate plan
def generate_plan(language):
    plan_prompt = PromptTemplate(
        template=plan_template,
        input_variables=["skill", "category", "days_available", "daily_time", "language"]
    )
    plan_chain = plan_prompt | gemini_model

    # Retry logic in case of API errors
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = plan_chain.invoke({
                "skill": skill,
                "category": selected_category,
                "days_available": days_available,
                "daily_time": daily_time,
                "language": language,
            })
            return response.content.strip() if hasattr(response, "content") else str(response).strip()
        
        except google.api_core.exceptions.InternalServerError as e:
            st.warning(f"⚠️ Server error: {e}. Retrying ({attempt+1}/3)...")
            time.sleep(3)  # Wait for 3 seconds before retrying
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            return None  # Stop execution if another error occurs

    st.error("❌ Failed to generate a plan after multiple attempts. Please try again later.")
    return None

# Generate plan button
if st.button("Generate Plan"):
    if not skill:
        st.error("Please enter a skill to master.")
    else:
        with st.spinner(f"📋 Generating plan in {selected_language}..."):
            plan = generate_plan(selected_language)
            st.markdown(f"## 🌍 Plan in {selected_language}:")
            st.markdown(plan)

test_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY
)

try:
    response = test_model.invoke("Hello! Can you respond?")
    print(response.content)
except Exception as e:
    print(f"API Error: {e}")

