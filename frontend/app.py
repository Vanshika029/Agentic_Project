import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Enterprise HR AI | Workforce Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Vanilla CSS with rich aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1E2F 0%, #2D1B69 50%, #111827 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }

    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Data Loader
@st.cache_data
def load_master_data():
    master_path = Path("data/processed/employee_intelligence_master.csv")
    gaps_path = Path("data/processed/organization_skill_gaps.csv")
    skills_path = Path("data/processed/employee_skills_inventory.csv")
    
    if master_path.exists():
        df_master = pd.read_csv(master_path)
    else:
        df_master = pd.DataFrame()
        
    if gaps_path.exists():
        df_gaps = pd.read_csv(gaps_path)
    else:
        df_gaps = pd.DataFrame()
        
    if skills_path.exists():
        df_skills = pd.read_csv(skills_path)
    else:
        df_skills = pd.DataFrame()
        
    return df_master, df_gaps, df_skills

df_master, df_gaps, df_skills = load_master_data()

# Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">⚡ Enterprise HR AI Platform</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 1.05rem;">
                Autonomous Workforce Intelligence • Attrition Risk Prediction • Skill Gap Engine • Upskilling Recommendation
            </p>
        </div>
        <div style="text-align: right; background: rgba(255,255,255,0.08); padding: 0.75rem 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 0.8rem; opacity: 0.7; text-transform: uppercase; font-weight: 700;">System Status</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #10b981;">● Online (v1.0)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.markdown("### 🎛️ Workforce Filters")
departments = ["All Departments"] + sorted(df_master['Department'].dropna().unique().tolist()) if not df_master.empty else ["All"]
selected_dept = st.sidebar.selectbox("Filter Department", departments)

job_roles = ["All Roles"]
if selected_dept != "All Departments" and not df_master.empty:
    job_roles += sorted(df_master[df_master['Department'] == selected_dept]['JobRole'].dropna().unique().tolist())
elif not df_master.empty:
    job_roles += sorted(df_master['JobRole'].dropna().unique().tolist())
selected_role = st.sidebar.selectbox("Filter Job Role", job_roles)

risk_filter = st.sidebar.multiselect("Filter Risk Level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])

# Apply Filters
filtered_df = df_master.copy()
if selected_dept != "All Departments":
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
if selected_role != "All Roles":
    filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
if risk_filter:
    filtered_df = filtered_df[filtered_df['RiskLevel'].isin(risk_filter)]

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

total_emp = len(filtered_df)
high_risk_count = int((filtered_df['RiskLevel'] == 'HIGH').sum()) if not filtered_df.empty else 0
high_risk_pct = round((high_risk_count / total_emp * 100), 1) if total_emp > 0 else 0
avg_engagement = round(float(filtered_df['EngagementScore'].mean()), 1) if not filtered_df.empty else 0
avg_prob = round(float(filtered_df['AttritionProbability'].mean() * 100), 1) if not filtered_df.empty else 0

with col1:
    st.metric("Total Workforce", f"{total_emp:,}", delta="Active Employees")
with col2:
    st.metric("High Risk Cohort", f"{high_risk_count}", delta=f"{high_risk_pct}% of team", delta_color="inverse")
with col3:
    st.metric("Avg Engagement", f"{avg_engagement}%", delta="+2.4% vs benchmark")
with col4:
    st.metric("Avg Attrition Risk", f"{avg_prob}%", delta="-1.1% past 30d")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_gaps, tab_recommendations, tab_employee_360, tab_simulator = st.tabs([
    "📊 Executive Overview",
    "🎯 Organization Skill Gaps",
    "💡 Upskilling Recommendations",
    "👤 Employee 360 Dossier",
    "🧪 What-If Simulator"
])

# Tab 1: Executive Overview
with tab_overview:
    c_left, c_right = st.columns([6, 5])
    
    with c_left:
        st.subheader("Departmental Attrition Risk Distribution")
        if not df_master.empty:
            dept_risk = df_master.groupby(['Department', 'RiskLevel']).size().unstack(fill_value=0).reset_index()
            for col in ['HIGH', 'MEDIUM', 'LOW']:
                if col not in dept_risk.columns:
                    dept_risk[col] = 0
            
            fig = px.bar(
                dept_risk,
                x='Department',
                y=['LOW', 'MEDIUM', 'HIGH'],
                title="Risk Breakdown by Department",
                color_discrete_map={'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'},
                barmode='stack'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans",
                legend_title_text='Risk Tier',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with c_right:
        st.subheader("Workforce Engagement vs Attrition Probability")
        if not df_master.empty:
            fig_scatter = px.scatter(
                filtered_df.sample(min(300, len(filtered_df))),
                x='EngagementScore',
                y='AttritionProbability',
                color='RiskLevel',
                color_discrete_map={'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'},
                hover_data=['EmployeeID', 'JobRole', 'Department'],
                title="Engagement vs Predicted Flight Risk"
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 2: Organization Skill Gaps
with tab_gaps:
    st.subheader("Critical Organization Skill Gaps")
    st.write("Calculated across the enterprise via the Set-Subtraction Skill Engine against O*NET Competency Standards.")
    
    if not df_gaps.empty:
        col_g1, col_g2 = st.columns([7, 4])
        with col_g1:
            fig_gap = px.bar(
                df_gaps.head(15),
                x='MissingEmployeeCount',
                y='Skill',
                orientation='h',
                color='Severity',
                color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
                title="Top 15 Enterprise Competency & Software Gaps"
            )
            fig_gap.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans"
            )
            st.plotly_chart(fig_gap, use_container_width=True)
            
        with col_g2:
            st.markdown("#### Severity Breakdown Table")
            st.dataframe(
                df_gaps.head(12)[['Skill', 'MissingEmployeeCount', 'Severity']],
                use_container_width=True,
                height=450
            )

# Tab 3: Upskilling Recommendations
with tab_recommendations:
    st.subheader("AI-Driven Learning Pathways & Upskilling Recommendations")
    st.write("Personalized course assignments mapped to close individual employee skill deficits.")
    
    if not filtered_df.empty:
        rec_table = filtered_df[['EmployeeID', 'Department', 'JobRole', 'RiskLevel', 'SkillGaps', 'UpskillingRecommendation']]
        st.dataframe(
            rec_table,
            use_container_width=True,
            height=500
        )

# Tab 4: Employee 360 Dossier
with tab_employee_360:
    st.subheader("Employee 360 Workforce Dossier")
    
    emp_ids = sorted(df_master['EmployeeID'].tolist()) if not df_master.empty else []
    selected_id = st.selectbox("Select Employee by ID", emp_ids, index=0 if emp_ids else None)
    
    if selected_id is not None and not df_master.empty:
        emp_record = df_master[df_master['EmployeeID'] == selected_id].iloc[0]
        
        c_p1, c_p2, c_p3 = st.columns([3, 4, 5])
        
        with c_p1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0 0 10px 0;">Employee #{emp_record['EmployeeID']}</h3>
                <p style="margin: 4px 0;"><strong>Role:</strong> {emp_record['JobRole']}</p>
                <p style="margin: 4px 0;"><strong>Dept:</strong> {emp_record['Department']}</p>
                <p style="margin: 4px 0;"><strong>Age:</strong> {emp_record['Age']} yrs</p>
                <p style="margin: 4px 0;"><strong>Salary:</strong> ${emp_record['MonthlyIncome']:,.2f}/mo</p>
                <hr style="opacity: 0.2;">
                <p style="margin: 4px 0;"><strong>Engagement Score:</strong> {emp_record['EngagementScore']}/100</p>
                <p style="margin: 4px 0;"><strong>Performance Rating:</strong> {emp_record['PerformanceRating']}/4</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c_p2:
            st.markdown("#### Attrition Risk Gauge")
            prob = float(emp_record['AttritionProbability'])
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk: {emp_record['RiskLevel']}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if prob >= 0.6 else ("#f59e0b" if prob >= 0.3 else "#10b981")},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.15)"},
                        {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.15)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c_p3:
            st.markdown("#### 🔍 Primary Attrition Drivers (Local SHAP)")
            drivers = str(emp_record['TopRiskDrivers']).split(", ")
            for d in drivers:
                st.markdown(f"• **{d}**")
                
            st.markdown("#### 🎯 Identified Skill Gaps")
            gaps = str(emp_record['SkillGaps']).split(", ")
            if gaps and gaps[0]:
                for g in gaps:
                    st.markdown(f"- ❌ *Missing:* `{g}`")
            else:
                st.success("No current skill deficits identified.")
                
            st.markdown("#### 🚀 Targeted Upskilling Path")
            st.info(emp_record['UpskillingRecommendation'])

# Tab 5: What-If Simulation
with tab_simulator:
    st.subheader("Real-Time Attrition Prediction & What-If Simulator")
    st.write("Adjust employee parameters to evaluate flight risk changes in real-time.")
    
    with st.form("simulation_form"):
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            sim_age = st.slider("Age", 18, 65, 32)
            sim_dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            sim_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])
            sim_income = st.number_input("Monthly Income ($)", min_value=1000.0, max_value=25000.0, value=5000.0, step=250.0)
        
        with s_col2:
            sim_ot = st.selectbox("OverTime", ["Yes", "No"], index=1)
            sim_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            sim_wlb = st.slider("Work Life Balance (1-4)", 1, 4, 3)
            sim_env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            
        with s_col3:
            sim_tenure = st.slider("Years At Company", 0, 35, 4)
            sim_role_tenure = st.slider("Years In Current Role", 0, 25, 2)
            sim_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
            sim_total_exp = st.slider("Total Working Years", 0, 40, 8)
            
        submit_btn = st.form_submit_button("🚀 Run On-Demand AI Prediction", use_container_width=True)
        
        if submit_btn:
            payload = {
                "EmployeeID": 9999,
                "Age": sim_age,
                "Department": sim_dept,
                "JobRole": sim_role,
                "MonthlyIncome": sim_income,
                "OverTime": sim_ot,
                "JobSatisfaction": sim_satisfaction,
                "WorkLifeBalance": sim_wlb,
                "EnvironmentSatisfaction": sim_env_sat,
                "RelationshipSatisfaction": 3,
                "YearsAtCompany": sim_tenure,
                "YearsInCurrentRole": sim_role_tenure,
                "YearsSinceLastPromotion": sim_promo,
                "TotalWorkingYears": sim_total_exp,
                "JobLevel": 2,
                "JobInvolvement": 3,
                "DistanceFromHome": 5
            }
            
            # Predict locally via predictor
            from app.ml.predictor import AttritionPredictor
            predictor = AttritionPredictor()
            res = predictor.predict_single(payload)
            
            st.success("✅ Prediction Generated Successfully!")
            st_c1, st_c2 = st.columns(2)
            with st_c1:
                st.metric("Predicted Attrition Probability", f"{res['AttritionProbability']*100:.1f}%")
                st.metric("Assigned Risk Level", res['RiskLevel'])
            with st_c2:
                st.write("Top Decision Drivers:")
                for dr in res['TopRiskDrivers']:
                    st.write(f"- ⚡ {dr}")
