import streamlit as st
import joblib
import json
import numpy as np
from bs4 import BeautifulSoup
import base64
import streamlit.components.v1 as components
from PIL import Image
import os

# Load model and metadata
model = joblib.load('ecg_decision_tree.pkl')
with open('ecg_feature_names.json', 'r') as f:
    feature_names = json.load(f)
with open('label_reverse_map.json', 'r') as f:
    label_map = json.load(f)

st.title("ECG Signal Classification App")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 EDA", "📖 Model Explanation"])

# Tab 1: Prediction
with tab1:
    st.header("Predict ECG Signal Type")
    st.write("Enter ECG measurements to predict the signal type (ARR, AFF, CHF, NSR):")

    user_input = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", step=0.01, format="%.4f")
        user_input.append(value)

    if st.button("Predict"):
        input_array = np.array(user_input).reshape(1, -1)
        pred = model.predict(input_array)[0]
        pred_label = label_map[str(pred)]
        st.success(f"Predicted ECG Signal Type: **{pred_label}**")
    
        
        image_path = "ecg_images/"    
        if pred_label == "ARR":
            img = Image.open(os.path.join(image_path, "ARR.png"))
        elif pred_label == "AFF":
            img = Image.open(os.path.join(image_path, "AFF.png"))
        elif pred_label == "CHF":
            img = Image.open(os.path.join(image_path, "CHF.png"))
        elif pred_label == "NSR":
            img = Image.open(os.path.join(image_path, "NSR.png"))

        st.image(img, caption=f"ECG Signal Type: {pred_label}", use_container_width=True)


with tab2:
    st.header("Exploratory Data Analysis (EDA) - Visuals Only")

    with open("ECG.html", "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src", "")
        if src.startswith("data:image"):
            base64_data = src.split(",")[1]
            image_bytes = base64.b64decode(base64_data)
            st.image(image_bytes, use_container_width=True)  


# Tab 3: Model Explanation and Summary
with tab3:
    st.header("Model Explanation and Summary")

    # Display the summary text with images to explain the columns
    st.markdown("""
    ## Summary of the Work Done

    This project focuses on **ECG signal classification**, where we trained a **Decision Tree** model to predict the type of ECG signal. The classification involves four distinct categories of ECG signals: 
    - **ARR** (Atrial Arrhythmia)
    - **AFF** (Atrial Fibrillation)
    - **CHF** (Congestive Heart Failure)
    - **NSR** (Normal Sinus Rhythm)

    ### Explanation of the Features Used in the Model

    The model uses the following features from the dataset:
    - **PQdis, QTdis, STdis, hbpermin, QRdis, RSdis**: These represent various measurements of the ECG signal that describe its characteristics in terms of time and amplitude.
    - **QRSseg, QRseg, STseg, Tseg, QTseg, Pseg**: These are segment-wise characteristics of the ECG waveform, corresponding to different phases of the heart's electrical activity.
    - **PQseg, QRslope, RSslope, STslope, PQslope, NNTot**: Additional segment-wise measurements and slopes that help capture the dynamics of the ECG signal, particularly the rate of change in different sections.
    
    """)

    # Image path
    image_path = "ecg_images/"    #the name of the folder that contain images
    img = Image.open(os.path.join(image_path, "all.jpg"))#the name of the  selected image
    st.image(img, use_container_width=True)  #to show image with frame

    st.markdown("""

    ### Explanation of the Columns in the Dataset:

    - **ECG_signal**: The target variable in the dataset, which contains the types of ECG signals. It is mapped to numerical values as follows:
        - **0**: ARR (Atrial Arrhythmia)
        - **1**: AFF (Atrial Fibrillation)
        - **2**: CHF (Congestive Heart Failure)
        - **3**: NSR (Normal Sinus Rhythm)
    
    These are the four possible classes for classification, and the model predicts one of these based on the feature inputs.

    ### Step-by-Step Data Preprocessing:
    - **Step 1**: Selection of relevant columns from the dataset:
        - Columns such as `PQdis`, `QTdis`, `STdis`, `hbpermin`, etc., were selected as they represent meaningful ECG features.
    - **Step 2**: Feature Engineering:
        - **PRseg** was created by summing `PQseg` and `QRseg` to capture combined information from those segments.
        - **PRdis** was created by adding `PQdis` and `QRdis`, and **QRSdis** was created by summing `QRdis` and `RSdis`.
        - Unused columns like `QRslope`, `RSslope`, and `PQdis` were dropped as they were considered redundant or irrelevant.
    - **Step 3**: Data Transformation:
        - The categorical values in the `ECG_signal` column were mapped to numerical values (`ARR`: 0, `AFF`: 1, `CHF`: 2, `NSR`: 3).
        - The dataset was cleaned and prepared for the model by ensuring all data types were correct and features were properly formatted.
    
    ### Model Training:
    - A **Decision Tree** model was trained on the processed dataset, using the selected features to predict the target variable (`ECG_signal`).
    - The model was evaluated using accuracy and confusion matrix, which were presented during the prediction tab to show how well the model performed on test data.

    ### How the Prediction Works:
    - When the user inputs the ECG measurements for each feature, the data is passed through the trained **Decision Tree** model.
    - The model then predicts the corresponding type of ECG signal (either ARR, AFF, CHF, or NSR) based on the input values.

    ## Model Accuracy and Performance
    The performance of the model is evaluated using accuracy on a test dataset, and the confusion matrix helps visualize the model’s predictions vs actual values.

    ### Confusion Matrix
    The confusion matrix allows us to understand how well the model is classifying each type of ECG signal by showing the true positive, false positive, true negative, and false negative counts for each class.

    """)
