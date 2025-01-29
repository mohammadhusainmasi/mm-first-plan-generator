from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time
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

📌 **Now, create a step-by-step plan** to help achieve mastery in **{skill}** within **{days_available}** days, dedicating **{daily_time}** hours daily. Ensure each day has unique tasks and maintains momentum toward the goal.
"""

# Initialize PromptTemplate
plan_prompt = PromptTemplate(
    template=plan_template,
    input_variables=["skill", "category", "days_available", "daily_time", "language"]
)

# Initialize Gemini Model
gemini_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Create the chain
plan_chain = plan_prompt | gemini_model

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
language = st.selectbox("Select language", ["English", "Spanish", "French", "German", "Hindi"])

# Generate plan button
if st.button("Generate Plan"):
    if not skill:
        st.error("Please enter a skill to master.")
    else:
        with st.spinner("📋 Generating your personalized plan..."):
            raw_plan = plan_chain.invoke({
                "skill": skill,
                "category": selected_category,
                "days_available": days_available,
                "daily_time": daily_time,
                "language": language,
            })
            plan = raw_plan.strip()

        # Display the plan
        st.markdown("### Your Personalized Plan:")
        st.markdown(plan)
