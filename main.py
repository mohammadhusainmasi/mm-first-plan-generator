from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_together import Together
from dotenv import load_dotenv
import os
import time
import streamlit as st

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

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
plan_template = f"""
🎯 **Your Task**:  
You are a **senior {{category}} professional**, renowned for your expertise and ranked in the **top 1%** of the market. Using your vast knowledge and experience, generate a **personalized plan** to master **{{skill}}** in the category of **{{category}}**, creating a significant impact by ensuring daily measurable progress.  

📅 **Plan Details**:  
1️⃣ **Skill to Master**: {{skill}}  
2️⃣ **Category**: {{category}}  
3️⃣ **Total Days Available**: {{days_available}}  
4️⃣ **Daily Time Commitment**: {{daily_time}} hours  
5️⃣ **Language**: {{language}}

📝 **Plan Requirements**:  
✅ Each day MUST include a **clear, actionable objective** with subtopics to master. No days should be skipped or merged into intervals.  
✅ Allocate **specific activities** for every single day, ensuring equal focus on **learning**, **practicing**, and **reviewing**.  
✅ Include milestones as **additional tasks** (not skipping days) for motivation and tracking.  
✅ Focus on practical learning with examples and exercises to solidify knowledge.  
✅ Use **emojis** and engaging language to make the plan approachable and fun.  

✨ **Example Plan**:  
📖 **Day 1**: Understand the basic concepts of {{skill}} (e.g., foundational terms, definitions).  
💡 **Day 2**: Practice foundational exercises (e.g., solve simple problems related to {{skill}}).  
📖 **Day 3**: Explore advanced subtopics of {{skill}} (e.g., advanced terms, implementation).  
...  
🚀 **Milestone Days**: Celebrate achievements and reinforce learning by completing milestone tasks (e.g., mini-projects or tests).  

📌 **Now, create a step-by-step plan** to help achieve mastery in **{{skill}}** within **{{days_available}}** days, dedicating **{{daily_time}}** hours daily. Ensure each day has unique tasks and maintains momentum toward the goal.
"""



# Initialize PromptTemplate with category and language
plan_prompt = PromptTemplate(
    template=plan_template,
    input_variables=["skill", "category", "days_available", "daily_time", "milestone_interval", "language"]
)

# Initialize models
gemini_model = GoogleGenerativeAI(model="gemini-1.0-pro")
mistral_model = Together(model="mistralai/Mistral-7B-Instruct-v0.3")
llama_model = Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
qwen_model = Together(model="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create chains for each model
plan_chains = {
    "Gemini Pro": plan_prompt | gemini_model,
    "Mistral 7B": plan_prompt | mistral_model,
    "LLaMA 70B": plan_prompt | llama_model,
    "Qwen 32B": plan_prompt | qwen_model,
}

# Streamlit UI
st.title("Personal Plan Generator")
st.subheader("Generate a step-by-step plan to master a skill using Generative AI")

# Model selection
selected_model = st.selectbox(
    "Select AI Model",
    options=list(plan_chains.keys()),
    help="Choose the AI model to generate your personalized plan."
)

# Category selection
selected_category = st.selectbox(
    "Select a Category",
    options=categories,
    help="Choose a category related to the skill you want to master."
)

# Input fields
skill = st.text_input("Enter the skill you want to master", placeholder="E.g., Python programming",   help="Specify the skill you want to master, such as programming, painting, or fitness.")
days_available = st.number_input(
    "Number of days available", min_value=1, max_value=365, value=10, step=1,
    help="Enter the total number of days you can dedicate to mastering the skill."
)
daily_time = st.number_input(
    "Daily time commitment (hours)", min_value=1, max_value=24, value=2, step=1,
    help="Specify how many hours you can commit to learning daily."
)
milestone_interval = st.number_input(
    "Milestone interval (days)", min_value=1, max_value=30, value=5, step=1,
    help="Set the interval for tracking progress, such as every 5 or 10 days."
)
language = st.selectbox(
    "Select language",
    ["English", "Gujarati", "Spanish", "French", "German", "Hindi", "Chinese", "Japanese", "Korean"], index=0
)


# Generate plan button
if st.button("Generate Plan"):
    if not skill:
        st.error("Please enter a skill to master.")
    else:
        with st.spinner(f"📋 Generating your personalized plan using {selected_model}..."):
            raw_plan = plan_chains[selected_model].invoke({
                "skill": skill,
                "category": selected_category,
                "days_available": days_available,
                "daily_time": daily_time,
                "milestone_interval": milestone_interval,
                "language": language,
            })
            plan = raw_plan.strip()

        # Success message
        success_placeholder = st.empty()
        success_placeholder.success("✨ Plan generated successfully!")
        time.sleep(0.2)
        success_placeholder.empty()

        # Display the plan
        st.markdown("### Your Personalized Plan:")
        # st.code(plan, language="text")
        st.markdown(plan)