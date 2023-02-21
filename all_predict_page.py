import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('all_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regr = data["model"]
sc_age = data["sc_age"]
sc_size = data["sc_size"]
le_ER = data["le_ER"]
le_PR = data["le_PR"]
le_HER2 = data["le_HER2"]


def show_predict_page_all():
    st.title("Predicting Survival of Breast Cancer Patients from Clinical data")
    st.write("""### We need some information to predict survival""")

    ER_STATUS = (
        "Negative",
        "Positive"
    )

    PR_STATUS = (
        "Negative",
        "Positive"
    )

    HER2_STATUS = (
        "Negative",
        "Positive"
    )

    AGE = st.slider("Age at diagnosis in years: ", 0, 100, 60)
    st.write('You selected:', AGE)

    TUMOR_SIZE = st.slider("Tumor size in millimeters: ", 0, 1000, 30)
    st.write('You selected:', TUMOR_SIZE)   

    ER_STATUS = st.selectbox('Estrogene receptor status: ', ER_STATUS)
    st.write('You selected:', ER_STATUS)

    PR_STATUS = st.selectbox('Progesterone receptor status: ', PR_STATUS)
    st.write('You selected:', PR_STATUS)

    HER2_STATUS = st.selectbox('Estrogen receptor status: ', HER2_STATUS)
    st.write('You selected:', HER2_STATUS)

    ok = st.button("Calculate Survival")
    if ok:
        X = np.array([[AGE, TUMOR_SIZE, ER_STATUS, PR_STATUS, HER2_STATUS]])

        X[:,0] = sc_age.transform(X[:,0].reshape(1, -1))
        X[:,1] = sc_size.transform(X[:,1].reshape(1, -1))
        X[:,2] = le_ER.transform(X[:,2])
        X[:,3] = le_PR.transform(X[:,3])
        X[:,4] = le_HER2.transform(X[:,4])
        X = X.astype(float)

        surv = regr.predict(X)
        st.subheader(f"The estimated survival in months: {surv[0]}")
