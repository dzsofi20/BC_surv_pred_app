import streamlit as st
from predict_page_ import show_predict_page
from explore_page_ import show_explore_page

page = st.sidebar.selectbox("Explore or Predict from different input data", ("Survival prediction", "Explore dataset"))

if page == "Survival prediction":
    show_predict_page()
else:
    show_explore_page()
