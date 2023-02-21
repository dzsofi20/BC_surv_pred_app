import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# @st.cache to do preprocessing only once
def load_data():
    df = pd.read_excel('seer_only_final.xlsx', sheet_name='final', header=0)
    return df
    
df = load_data()

def show_explore_page():
    st.title("Eplore SEER data")
    st.write("""SEER data""")

    data = df.groupby(["AGE_AT_DIAGNOSIS_years"])["SURVIVAL_months"].mean().sort_values(ascending=True)
    st.bar_chart(data)