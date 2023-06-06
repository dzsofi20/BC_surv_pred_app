import streamlit as st
import pickle
import numpy as np
import bz2file as bz2

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

data = decompress_pickle('steps.pbz2')
regr = data["model"]
le_hist = data["le_hist"]
le_site = data["le_site"]
le_ER = data["le_ER"]
le_PR = data["le_PR"]
le_HER2 = data["le_HER2"]
le_inc = data["le_inc"]
le_gr = data["le_gr"]
le_st = data["le_st"]
le_surg = data["le_surg"]
le_bone = data["le_bone"]
le_liver = data["le_liver"]
le_brain = data["le_brain"]
sc_age = data["sc_age"]
sc_size = data["sc_size"]
sc_mal = data["sc_mal"]
sc_exa = data["sc_exa"]
sc_pos = data["sc_pos"]


def show_predict_page():
    st.title("Predicting Survival of Breast Cancer Patients from Clinical data")
    st.write("""### We need some information to predict survival""")

    hist_subtype = (
        "8500", "8541", "8522", "8523", "8520", "8480", "8575", "8507", "8524", "8050", "8530", "8504", "8343", "8032",
        "8540", "8501", "8401", "8510", "8503", "8323", "8140", "8521", "8200", "8074", "8070", "8033", "8201", "8000",
        "8022", "8255", "8010", "8211", "8246", "8315", "8490", "8041", "8982", "8983", "8013", "8543", "8230", "8004",
        "8574", "8020", "8572", "8123", "8021", "8260", "8513", "8571", "8071", "8570", "8310", "8560", "8550", "8720"
    )

    primary_site = (
        'C50.6-Axillary tail of breast', 'C50.1-Central portion of breast', 'C50.3-Lower-inner quadrant of breast', 'C50.9-Breast, NOS',
        'C50.8-Overlapping lesion of breast', 'C50.4-Upper-outer quadrant of breast', 'C50.2-Upper-inner quadrant of breast', 'C50.5-Lower-outer quadrant of breast', 'C50.0-Nipple'
    )

    ER_status = (
        'Negative', 'Positive', 'Borderline'
    )

    PR_status = (
        'Negative', 'Positive', 'Borderline'
    )

    HER2_status = (
        'Negative', 'Positive', 'Borderline'
    )

    income = (
        '$75,000+', '$70,000 - $74,999', '$60,000 - $64,999', '$65,000 - $69,999', '$40,000 - $44,999', '$45,000 - $49,999', '$55,000 - $59,999',
        '$50,000 - $54,999', '$35,000 - $39,999', '< $35,000', 'Unknown/missing/no match/Not 1990-2018'
    )

    grade = (
        'Poorly differentiated; Grade III', 'Moderately differentiated; Grade II', 'Well differentiated; Grade I', 'Undifferentiated; anaplastic; Grade IV', 'Unknown'
    )

    stage = (
        'Localized', 'Regional', 'Distant', 'Unknown/unstaged'
    )

    surgery = (
        'No', 'Yes'
    )

    bone = (
        'No', 'Yes', 'Unknown'
    )

    liver = (
        'No', 'Yes', 'Unknown'
    )

    brain = (
        'No', 'Unknown', 'Yes'
    )

    AGE = st.slider("Age at diagnosis in years: ", 0, 100, 60)
    st.write('You selected:', AGE)

    TUMOR_SIZE = st.slider("Tumor size in millimeters: ", 0, 1000, 30)
    st.write('You selected:', TUMOR_SIZE)   

    HIST_TYPE = st.selectbox("Histologic type (based on ICD-O-3): ", hist_subtype)
    st.write('You selected:', HIST_TYPE)
    
    PRIM_SITE = st.selectbox("Primary site: ", primary_site)
    st.write('You selected:', PRIM_SITE)

    ER_STATUS = st.selectbox("ER status: ", ER_status)
    st.write('You selected:', ER_STATUS)

    PR_STATUS = st.selectbox("PR status: ", PR_status)
    st.write('You selected:', PR_STATUS)

    HER2_STATUS = st.selectbox("HER2 status: ", HER2_status)
    st.write('You selected:', HER2_STATUS)

    INCOME = st.selectbox("Yearly income: ", income)
    st.write('You selected:', INCOME)

    GRADE = st.selectbox("Grade of tumor: ", grade)
    st.write('You selected: ', GRADE)

    STAGE = st.selectbox("Stage: ", stage)
    st.write('You selected: ', STAGE)

    SURGERY = st.selectbox("Surgery performed: ", surgery)
    st.write('You selected: ', SURGERY)

    BONE_M = st.selectbox("Bone metastases: ", bone)
    st.write('You selected: ', BONE_M)

    LIVER_M = st.selectbox("Liver metastases: ", liver)
    st.write('You selected: ', LIVER_M)

    BRAIN_M = st.selectbox("Brain metastases: ", brain)
    st.write('You selected: ', BRAIN_M)

    MALS = st.slider("Total number of malignant tumors: ", 0, 10, 3)
    st.write('You selected:', MALS)   

    EX_NODES = st.slider("Regional nodes examined: ", 0, 90, 10)
    st.write('You selected: ', EX_NODES)

    POS_NODES = st.slider("Regional nodes positive: ", 0, 90, 10)
    st.write('You selected: ', POS_NODES)

    ok = st.button("Calculate Survival")
    if ok:
        X = np.array([[AGE, HIST_TYPE, ER_STATUS, PR_STATUS, HER2_STATUS, PRIM_SITE, SURGERY, INCOME, GRADE, STAGE, MALS, EX_NODES, POS_NODES, BONE_M, LIVER_M, BRAIN_M, TUMOR_SIZE]])

        X[:,0] = sc_age.transform(X[:,0].reshape(1, -1))
        X[:,1] = le_hist.transform(X[:,1])
        X[:,2] = le_ER.transform(X[:,2])
        X[:,3] = le_PR.transform(X[:,3])
        X[:,4] = le_HER2.transform(X[:,4])
        X[:,5] = le_site.transform(X[:,5])
        X[:,6] = le_surg.transform(X[:,6])
        X[:,7] = le_inc.transform(X[:,7])
        X[:,8] = le_gr.transform(X[:,8])
        X[:,9] = le_st.transform(X[:,9])
        X[:,10] = sc_mal.transform(X[:,10].reshape(1, -1))
        X[:,11] = sc_exa.transform(X[:,11].reshape(1, -1))
        X[:,12] = sc_pos.transform(X[:,12].reshape(1, -1))
        X[:,13] = le_bone.transform(X[:,13])
        X[:,14] = le_liver.transform(X[:,14])
        X[:,15] = le_brain.transform(X[:,15])
        X[:,16] = sc_size.transform(X[:,16].reshape(1, -1))
        X = X.astype(float)

        surv = regr.predict(X)
        st.subheader(f"The estimated survival in months: {surv[0]}")
