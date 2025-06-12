import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os

# Load trained model
model = joblib.load("career_model.pkl")

# Define labels used during training
interest_labels = ['coding', 'analysis', 'leadership', 'people', 'building', 'logic',
                   'problem solving', 'planning', 'business', 'analytics', 'organizing',
                   'debugging', 'creativity', 'innovation', 'strategy', 'negotiation',
                   'system design', 'statistics', 'project management', 'marketing', 
                   'networking', 'design']

subject_labels = ['cs', 'math', 'economics', 'psychology', 'physics', 'stats', 'management',
                  'finance', 'english', 'business', 'it', 'art', 'design', 'ai', 'sociology']

# Fit encoders
mlb_interests = MultiLabelBinarizer(classes=interest_labels)
mlb_subjects = MultiLabelBinarizer(classes=subject_labels)
mlb_interests.fit([interest_labels])
mlb_subjects.fit([subject_labels])

# Career field skill profiles for comparison
career_skill_profiles = {
    "Data Science": {"Communication": 6, "Programming": 9, "Management": 5, "Creativity": 6},
    "Computer Science": {"Communication": 5, "Programming": 9, "Management": 4, "Creativity": 5},
    "Management Science": {"Communication": 9, "Programming": 4, "Management": 9, "Creativity": 6},
}

# Streamlit UI
st.set_page_config(page_title="Career Recommendation System", layout="centered")
st.title("🎓 Career Recommendation System")
st.markdown("**Get suggestions based on your interests, subjects, and skills.**")

# Multiselect inputs
interests = st.multiselect("Select your interests", interest_labels, default=["coding"])
subjects = st.multiselect("Select your favorite subjects", subject_labels, default=["cs"])

# Sliders for skills
comm = st.slider("🗣️ Communication Skills", 1, 10, 5)
prog = st.slider("💻 Programming Skills", 1, 10, 5)
mgmt = st.slider("📋 Management Skills", 1, 10, 5)
crea = st.slider("🎨 Creativity", 1, 10, 5)

# Predict & display
if st.button("Get Career Recommendation"):
    try:
        encoded_interests = mlb_interests.transform([interests])
        encoded_subjects = mlb_subjects.transform([subjects])
        input_features = pd.concat([
            pd.DataFrame(encoded_interests),
            pd.DataFrame(encoded_subjects),
            pd.DataFrame([[comm, prog, mgmt, crea]])
        ], axis=1)

        prediction = model.predict(input_features)[0]
        st.success(f"✅ **Recommended Field:** {prediction}")

    except ValueError as ve:
        st.error("❌ Invalid input. Please make sure your interests and subjects are from the provided list.")
        st.error(str(ve))

    else:
        # Match Score Comparison
        st.markdown("### 🔍 How You Match With Other Fields")
        match_scores = {}
        for field, profile in career_skill_profiles.items():
            diff = sum(abs(profile[skill] - score) for skill, score in zip(profile.keys(), [comm, prog, mgmt, crea]))
            score = 100 - diff * 5
            match_scores[field] = max(score, 0)

        for field, score in match_scores.items():
            st.markdown(f"**{field} Match:** 🌟 {score:.1f}%")
            with st.expander(f"🧩 {field} Profile vs Yours"):
                st.write(f"**Ideal Skills:** {career_skill_profiles[field]}")
                st.write(f"**Your Skills:** Communication={comm}, Programming={prog}, Management={mgmt}, Creativity={crea}")

        # Skill Bar Chart
        st.markdown("### 🧠 Your Skill Profile")
        skill_data = pd.DataFrame({
            "Skill": ["Communication", "Programming", "Management", "Creativity"],
            "Score": [comm, prog, mgmt, crea]
        })
        st.bar_chart(skill_data.set_index("Skill"))

        # Radar Chart (Plotly)
        st.markdown("### 📊 Skill Comparison Radar")

        user_skills = {"Communication": comm, "Programming": prog, "Management": mgmt, "Creativity": crea}
        ideal_skills = career_skill_profiles[prediction]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=list(user_skills.values()),
            theta=list(user_skills.keys()),
            fill='toself',
            name='Your Skills'
        ))

        fig.add_trace(go.Scatterpolar(
            r=list(ideal_skills.values()),
            theta=list(ideal_skills.keys()),
            fill='toself',
            name=f'{prediction} Ideal'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            title="🎯 Skill Radar: You vs Ideal for " + prediction
        )

        # Save and show radar chart
        fig_path = "skill_radar_chart.html"
        fig.write_html(fig_path)
        components.html(open(fig_path, "r").read(), height=500)

        # Career role suggestions
        st.markdown("## 🚀 Personalized Career Suggestions")
        career_map = {
            "Data Science": [
                {"title": "🔍 Data Analyst", "desc": "Clean, analyze, and visualize data for business decisions.",
                 "tools": "Excel, SQL, Python (Pandas), Power BI", "courses": "Google Data Analytics, IBM DA",
                 "salary": "💰 $70,000/year"},
                {"title": "🤖 ML Engineer", "desc": "Build smart systems that learn from data.",
                 "tools": "Python, Scikit-learn, TensorFlow", "courses": "ML by Andrew Ng, FastAI",
                 "salary": "💰 $110,000/year"},
                {"title": "📊 Data Scientist", "desc": "Extract knowledge from data with ML and stats.",
                 "tools": "Python, R, SQL, Tableau", "courses": "Coursera DS Specialization",
                 "salary": "💰 $120,000/year"}
            ],
            "Computer Science": [
                {"title": "💻 Software Engineer", "desc": "Develop apps, APIs, and systems.",
                 "tools": "Java, Python, Git, Linux", "courses": "Harvard CS50, DSA on Coursera",
                 "salary": "💰 $105,000/year"},
                {"title": "🌐 Web Developer", "desc": "Create websites and web apps.",
                 "tools": "HTML, CSS, JS, React", "courses": "The Odin Project, FreeCodeCamp",
                 "salary": "💰 $85,000/year"},
                {"title": "🔐 System Admin", "desc": "Manage IT infrastructure & servers.",
                 "tools": "Linux, Bash, Windows Server", "courses": "CompTIA Linux+, RHCSA",
                 "salary": "💰 $75,000/year"}
            ],
            "Management Science": [
                {"title": "📦 Product Manager", "desc": "Lead product development and teams.",
                 "tools": "JIRA, Trello, Figma", "courses": "PM Bootcamp, Coursera PM",
                 "salary": "💰 $115,000/year"},
                {"title": "📈 Business Analyst", "desc": "Analyze business needs and solutions.",
                 "tools": "Excel, SQL, Tableau", "courses": "Udemy BA, DataCamp",
                 "salary": "💰 $90,000/year"},
                {"title": "🧠 HR Specialist", "desc": "Manage recruitment and employee relations.",
                 "tools": "BambooHR, Zoho HR, Excel", "courses": "Coursera HRM, SHRM Prep",
                 "salary": "💰 $65,000/year"}
            ]
        }

        for role in career_map.get(prediction, []):
            with st.expander(role["title"]):
                st.markdown(f"**📝 Description:** {role['desc']}")
                st.markdown(f"**🛠️ Tools & Technologies:** {role['tools']}")
                st.markdown(f"**📚 Suggested Courses:** {role['courses']}")
                st.markdown(f"**💸 Average Salary:** {role['salary']}")