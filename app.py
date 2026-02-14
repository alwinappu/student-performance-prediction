import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Performance Prediction System")
st.markdown("""
### ML-Powered Academic Performance Analyzer
Predict student performance using **Random Forest**, **Gradient Boosting**, and **Logistic Regression**.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
**Project**: Student Performance Prediction  
**Modules**: 
- Data Preprocessing
- ML Model Training
- Performance Analytics
- Visualization Dashboard
""")

# Generate sample data function
@st.cache_data
def generate_sample_data(n_samples=200):
    np.random.seed(42)
    data = pd.DataFrame({
        'Attendance': np.random.randint(50, 100, n_samples),
        'Internal_Scores': np.random.randint(20, 100, n_samples),
        'Assignment_Submissions': np.random.randint(0, 15, n_samples),
        'Engagement_Level': np.random.randint(1, 5, n_samples),
        'Study_Hours_Per_Week': np.random.randint(5, 50, n_samples),
        'Lab_Work_Hours': np.random.randint(0, 40, n_samples),
        'Participation_Score': np.random.randint(0, 100, n_samples)
    })
    performance_score = (data['Attendance'] * 0.2 + data['Internal_Scores'] * 0.3 + 
                        data['Assignment_Submissions'] * 3 + data['Engagement_Level'] * 10 +
                        data['Study_Hours_Per_Week'] * 0.8 + data['Lab_Work_Hours'] * 0.5 +
                        data['Participation_Score'] * 0.2)
    data['Performance'] = (performance_score > np.percentile(performance_score, 40)).astype(int)
    return data

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Model Training", "📈 Predictions"])

with tab1:
    st.header("Dataset Overview")
    data = generate_sample_data()
    st.dataframe(data.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", len(data))
        st.metric("Pass Rate", f"{(data['Performance'].sum() / len(data) * 100):.1f}%")
    with col2:
        fig, ax = plt.subplots()
        data['Performance'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Performance Distribution')
        ax.set_xticklabels(['Fail', 'Pass'], rotation=0)
        st.pyplot(fig)

with tab2:
    st.header("Model Training & Evaluation")
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            data = generate_sample_data()
            X = data.drop('Performance', axis=1)
            y = data['Performance']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred)
                }
            
            st.success("✅ Models trained successfully!")
            st.dataframe(pd.DataFrame(results).T, use_container_width=True)
            
            # Best model
            best_model = max(results, key=lambda x: results[x]['F1-Score'])
            st.info(f"🏆 Best Model: **{best_model}** (F1-Score: {results[best_model]['F1-Score']:.4f})")

with tab3:
    st.header("Make Predictions")
    st.write("Enter student data to predict performance:")
    
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance %", 50, 100, 75)
        internal = st.slider("Internal Scores", 20, 100, 60)
        assignments = st.slider("Assignments Submitted", 0, 15, 10)
        engagement = st.slider("Engagement Level", 1, 5, 3)
    with col2:
        study_hours = st.slider("Study Hours/Week", 5, 50, 25)
        lab_hours = st.slider("Lab Work Hours", 0, 40, 20)
        participation = st.slider("Participation Score", 0, 100, 70)
    
    if st.button("Predict Performance"):
        # Simple prediction logic
        score = (attendance * 0.2 + internal * 0.3 + assignments * 3 + 
                engagement * 10 + study_hours * 0.8 + lab_hours * 0.5 + participation * 0.2)
        prediction = "Pass ✅" if score > 150 else "Fail ❌"
        st.success(f"Predicted Performance: **{prediction}**")
        st.metric("Confidence Score", f"{min(score/200 * 100, 100):.1f}%")

st.markdown("---")
st.caption("© 2026 Student Performance Prediction System | College Project")
