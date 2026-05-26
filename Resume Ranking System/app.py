# Resume Ranking & Job Matching System

import streamlit as st
import pandas as pd
import re
import pdfplumber

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# PAGE CONFIG

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# CLEANING FUNCTION

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.lower().strip()


# PDF TEXT EXTRACTION

def extract_text_from_pdf(uploaded_file):
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# SKILLS DATABASE

skills_list = [
    "python",
    "java",
    "c++",
    "machine learning",
    "deep learning",
    "sql",
    "power bi",
    "excel",
    "tableau",
    "nlp",
    "data analysis",
    "data science",
    "tensorflow",
    "pytorch",
    "html",
    "css",
    "javascript",
    "react",
    "nodejs",
    "mongodb",
    "docker",
    "kubernetes",
    "aws",
    "git",
    "github"
]


# SKILL EXTRACTION

def extract_skills(text):
    found_skills = []

    text = text.lower()

    for skill in skills_list:
        if skill.lower() in text:
            found_skills.append(skill)

    return list(set(found_skills))


# UI/Dashboard

st.title("📄Resume Ranking System")

st.write(
    "Upload multiple resumes and compare them against a job description."
)


# JOB DESCRIPTION INPUT

job_description = st.text_area(
    "Enter Job Description",
    height=250,
    placeholder="Example: Looking for a Python developer with Machine Learning and NLP skills..."
)


# MULTIPLE FILE UPLOAD

uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=['pdf', 'txt'],
    accept_multiple_files=True
)


# PROCESSING


if st.button("Rank Resumes"):

    if not job_description:
        st.warning("Please enter a job description.")

    elif not uploaded_files:
        st.warning("Please upload resumes.")

    else:

        resume_texts = []
        resume_names = []

        
        # READ RESUMES
        
        for uploaded_file in uploaded_files:

            try:

                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)

                else:
                    resume_text = uploaded_file.read().decode('utf-8')

                cleaned_resume = clean_text(resume_text)

                resume_texts.append(cleaned_resume)
                resume_names.append(uploaded_file.name)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        
        # TF-IDF + COSINE SIMILARITY

        all_documents = [job_description] + resume_texts

        tfidf = TfidfVectorizer(stop_words='english')

        tfidf_matrix = tfidf.fit_transform(all_documents)

        job_description_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        similarity_scores = cosine_similarity(
            job_description_vector,
            resume_vectors
        )

        scores = similarity_scores[0]

        # CREATE RESULTS
        

        results = []

        for i in range(len(resume_names)):

            match_percentage = round(scores[i] * 100, 2)

            resume_skills = extract_skills(resume_texts[i])
            jd_skills = extract_skills(job_description)

            matched_skills = list(
                set(resume_skills).intersection(set(jd_skills))
            )

            missing_skills = list(
                set(jd_skills) - set(resume_skills)
            )

            results.append({
                'Resume': resume_names[i],
                'Match %': match_percentage,
                'Matched Skills': ', '.join(matched_skills),
                'Missing Skills': ', '.join(missing_skills)
            })

        
        # SORT RESULTS
        
        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by='Match %',
            ascending=False
        )

        results_df.reset_index(drop=True, inplace=True)

        results_df.index = results_df.index + 1


        # DISPLAY RESULTS

        st.success("Resume ranking completed successfully!")

        st.subheader("🏆 Ranked Resumes")

        st.dataframe(results_df, use_container_width=True)


        # TOP CANDIDATE
        
        top_candidate = results_df.iloc[0]

        st.subheader("⭐ Best Matched Resume")

        st.write(f"Resume Name: {top_candidate['Resume']}")
        st.write(f"Match Percentage: {top_candidate['Match %']}%")

        
        # CHART
        
        chart_data = results_df[['Resume', 'Match %']]
        chart_data = chart_data.set_index('Resume')

        st.subheader("📊 Resume Match Scores")

        st.bar_chart(chart_data)


