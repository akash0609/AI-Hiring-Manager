import streamlit as st

from query import run_hiring_analysis

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="AI Hiring Manager",
    layout="wide"
)


# =====================================================
# Title
# =====================================================

st.title("AI Hiring Manager")


st.markdown("""
Enterprise RAG-based Hiring Assistant
""")


# =====================================================
# Input Role
# =====================================================

role = st.text_input(
    "Enter Hiring Role",
    value="AVP Collections Analytics"
)


# =====================================================
# Button
# =====================================================

if st.button("Analyze Candidates"):

    with st.spinner("Running AI Analysis..."):

        query = f"""
        Identify the best candidates for:
        {role}

        Focus on:
        - relevant experience
        - skills
        - exposure to relevant tools/technologies
        - banking analytics
        - SQL/Tableau skills
        - leadership
        - AI/automation exposure
        """

        result = run_hiring_analysis(query)

    st.success("Analysis Completed")


    # =================================================
    # Display Result
    # =================================================

    st.markdown("## Hiring Recommendation")

    st.write(result)