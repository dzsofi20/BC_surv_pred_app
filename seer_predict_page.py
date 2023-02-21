import streamlit as st
import pickle
import numpy as np
import bz2file as bz2

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

data = decompress_pickle('seer_steps.pbz2')
regr = data["model"]
sc_age = data["sc_age"]
sc_size = data["sc_size"]
sc_nodes = data["sc_nodes"]
le_neoadj = data["le_neoadj"]
le_subtype = data["le_subtype"]
le_stage = data["le_stage"]
le_surgery = data["le_surgery"]


def show_predict_page_seer():
    st.title("Predicting Survival of Breast Cancer Patients from Clinical data")
    st.write("""### We need some information to predict survival""")

    surgery = (
        "Yes",
        "No"
    )

    neoadj = (
        "Neoadjuvant therapy not given", 
        "Unknown",
        "Stated as partial response (PR)", 
        "Stated as no response (NR)",
        "Stated as complete response (CR)"
    )

    subtype = (
        "HR-/HER2-", 
        "HR+/HER2+", 
        "HR+/HER2-", 
        "HR-/HER2+"
    )

    stage = (
        "Regional", 
        "Localized", 
        "Distant"
    )

    AGE = st.slider("Age at diagnosis in years: ", 0, 85, 50)
    st.write('You selected:', AGE)

    SURGERY_PERFORMED = st.selectbox("Was surgery performed?: ", surgery)
    st.write('You selected:', SURGERY_PERFORMED)

    TUMOR_SIZE = st.slider("Tumor size in millimeters: ", 0, 998, 10)
    st.write('You selected:', TUMOR_SIZE)   

    NODES = st.slider("Regional nodes positive: ", 0, 55, 30)
    st.write('You selected:', NODES)

    NEOADJ = st.selectbox('What was the response to neoadjuvant treatment?', neoadj)
    st.write('You selected:', NEOADJ)

    SUBTYPE = st.selectbox('Breast cancer subtype', subtype)
    st.write('You selected:', SUBTYPE)

    STAGE = st.selectbox('Breast cancer stage', stage)
    st.write('You selected:', STAGE)

    ok = st.button("Calculate Survival")
    if ok:
        X = np.array([[AGE, TUMOR_SIZE, NODES, NEOADJ, SUBTYPE, STAGE, SURGERY_PERFORMED]])

        X[:,0] = sc_age.transform(X[:,0].reshape(1, -1))
        X[:,1] = sc_size.transform(X[:,1].reshape(1, -1))
        X[:,2] = sc_nodes.transform(X[:,2].reshape(1, -1))
        X[:,3] = le_neoadj.transform(X[:,3])
        X[:,4] = le_subtype.transform(X[:,4])
        X[:,5] = le_stage.transform(X[:,5])
        X[:,6] = le_surgery.transform(X[:,6])
        X = X.astype(float)

        surv = regr.predict(X)
        st.subheader(f"The estimated survival in months: {surv[0]}")
