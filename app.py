import streamlit as st
from seer_predict_page import show_predict_page_seer
from all_predict_page import show_predict_page_all
from explore_page import show_explore_page

page = st.sidebar.selectbox("Explore or Predict from different input data", ("Predict from only SEER data", "Predict from combined dataset" ,"Explore datasets"))

if page == "Predict from only SEER data":
    show_predict_page_seer()
elif page == "Predict from combined dataset":
    show_predict_page_all()
else:
    show_explore_page()