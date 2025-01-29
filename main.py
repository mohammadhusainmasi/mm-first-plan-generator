from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()

# API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Please check your .env file.")
    st.stop()

# Categories for selection
categories = [
    "Programming 💻",
    "Art 🎨",
    "Music 🎶",
    "Fitness 🏋️",
    "Cooking 🍳",
    "Languages 🌍",
    "Learning ✍️",
    "Business 📈"
]

# Default temperature settings for different languages
language_temperatures = {
    "English": 0.7,
    "Spanish": 0.8,
    "French": 0.75,
    "German": 0.6,
    "Hindi": 0.85,
    "Chinese": 0.65,
    "Japanese": 0.7,
    "Gujarati":0.1, 
}

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
st.title("Personal Plan Generator")
st.subheader("Generate a step-by-step plan to master a skill using Generative AI")

# Category selection
selected_category = st.selectbox(
    "Select a Category",
    options=categories,
    help="Choose a category related to the skill you want to master."
)

# Input fields
skill = st.text_input("Enter the skill you want to master", placeholder="E.g., Python programming")
days_available = st.number_input("Number of days available", min_value=1, max_value=365, value=10)
daily_time = st.number_input("Daily time commitment (hours)", min_value=1, max_value=24, value=2)

# Multi-language selection
available_languages = list(language_temperatures.keys())
selected_languages = st.multiselect("Select Output Languages", available_languages, default=["English"])

# Compute default temperature based on selected languages
if selected_languages:
    avg_temperature = sum(language_temperatures[lang] for lang in selected_languages) / len(selected_languages)
else:
    avg_temperature = 0.7  # Default to English

# Temperature slider (allows manual override)
temperature = st.slider(
    "Select Model Creativity (Temperature)", 
    min_value=0.0, max_value=1.0, 
    value=avg_temperature, step=0.1
)

# Initialize Gemini Model with dynamic temperature
gemini_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=temperature)

# Create the chain
def generate_plan_for_language(language):
    plan_prompt = PromptTemplate(
        template=plan_template,
        input_variables=["skill", "category", "days_available", "daily_time", "language"]
    )
    plan_chain = plan_prompt | gemini_model

    response = plan_chain.invoke({
        "skill": skill,
        "category": selected_category,
        "days_available": days_available,
        "daily_time": daily_time,
        "language": language,
    })
    return response.content.strip() if hasattr(response, "content") else str(response).strip()

# Generate plan button
if st.button("Generate Plan"):
    if not skill:
        st.error("Please enter a skill to master.")
    else:
        st.markdown("### Your Personalized Plans:")
        for lang in selected_languages:
            with st.spinner(f"📋 Generating plan in {lang}..."):
                plan = generate_plan_for_language(lang)
                st.markdown(f"## 🌍 Plan in {lang}:")
                st.markdown(plan)
