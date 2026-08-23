import streamlit as st 
from backend.detect_disease import detect
from backend.gemini import explain_disease
st.header(":green[Plant] Disease Detection",divider="blue")

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🪴",
    layout="wide",
)

ALLOWED_FILE_TYPES = ['png','jpg','jpeg']

st.markdown(
    '''
     In this application you can upload an image of plant leaf and you can get whether plant is healty or not.
    '''
)
st.write("Upload an image of plant leaf :)")

uploaded_file = st.file_uploader(
    "Upload your document",
    type=ALLOWED_FILE_TYPES,
    max_upload_size=2
)
if uploaded_file is not None:
    st.success(f"Uploaded:{uploaded_file.name}")

    submit = st.button("submit",type="primary")
    if submit :
        try:
            with st.spinner("processing.."):
                label, confidence = detect(uploaded_file)
                st.write("### Prediction")
                st.write(label)
                st.write(f"Confidence: {confidence:.2%}")

                explanation = explain_disease(label,confidence)
                st.write("### Explanation")
                st.markdown(explanation)
        except Exception as error:
            st.error(f"An error occured while processing"
                     f"Error:{error}")