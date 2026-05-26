import streamlit as st
import pickle
import re
import nltk
import pdfplumber

nltk.download('punkt')
nltk.download('stopwords')

# loading models

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    
    # remove URLs
    cleanTxt = re.sub(r'http\S+|www\S+', ' ', txt)
    
    # remove emails
    cleanTxt = re.sub(r'\S+@\S+', ' ', cleanTxt)
    
    # remove hashtags and mentions
    cleanTxt = re.sub(r'#\S+|@\S+', ' ', cleanTxt)
    
    # remove newline characters
    cleanTxt = re.sub(r'\n', ' ', cleanTxt)
    
    # remove special characters and punctuation
    cleanTxt = re.sub(r'[^A-Za-z0-9 ]+', ' ', cleanTxt)
    
    # remove extra spaces
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()
    
    return cleanTxt

# web app
def main():
    st.title("Resume Screening App")
    st.subheader("Upload resume to know the category")
    uploaded_file = st.file_uploader("Upload Resume", type= ['txt','pdf'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    resume_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            resume_text += text
            else:
                resume_text = uploaded_file.read().decode("utf-8")

        except UnicodeDecodeError:
            # if utf-8 fails try decoding with 'latin-1'
            resume_text = uploaded_file.read().decode("latin-1")
            
        cleaned_resume = cleanResume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)
        
        category_mapping = {
            0: "Business Analyst",
            1: "DevOps Engineer",
            2: "Finance",
            3: "Software Engineer",
            4: "Web Developer"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("category name: ", category_name)
    
# pyhthon main

if __name__ == "__main__" :
    main()