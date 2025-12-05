import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Main content window
st.markdown(
    "<h1 style='text-align: center;'>Can I predict if you use LinkedIn?</h1>",
    unsafe_allow_html=True
)

st.write ("---")
st.write(f"Click the sidebar to the left and when all your responses are updated click the predict button below.")

# Create a button
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
def on_button_click():
    st.session_state.button_clicked = True

# Just the button – CSS will center it
st.button('Predict!', on_click=on_button_click)


# Sidebar Content
st.sidebar.title("Tell Me About Yourself")
st.sidebar.write("Note: this data will not be saved or used outside of this app.")

name = st.sidebar.text_input("What should I call you?")

Income = st.sidebar.selectbox(
        "What is your household income?",
        ("Select income...","Less than $10,000", "10 to under $20,000", "20 to under $30,000", "30 to under $40,000", "40 to under $50,000", "50 to under $75,000", "75 to under $100,000", "100 to under $150,000", "Greater than $150,000"),
    index=0
)
## Income Conversion
if Income == "Select income...":
    st.sidebar.warning("Please select your household income.")
    st.stop()
if Income == "Less than $10,000":
    Income = 1
elif Income == "10 to under $20,000":
    Income = 2
elif Income == "30 to under $40,000":
    Income = 3
elif Income == "40 to under $50,000":
    Income = 4
elif Income == "50 to under $75,000":
    Income = 5
elif Income == "75 to under $100,000":
    Income = 6
elif Income == "100 to under $150,000":
    Income = 7
else: Income = 8
    
Education = st.sidebar.selectbox(
        "What is your highest level of education?",
        ("Select education level...","Less than High School (Grades 1-8 or no formal schooling)", "High School Incomplete (Grades 9-11 or Grade 12 with no diploma)", "High School Graduate (Grade 12 with diploma or GED certificate)", "Some college, no degree (includes some community college)", "Two-year associate degree from a college or university", "Four-year college or university degree/Bachelor's degree", "Some postgraduate or professional schooling, no post-grad degree", "Postgraduate or professional degree, including masters, doctorate, medical or law degree"),
    index=0
)

## Education Conversion
if Education == "Select education level...":
    st.sidebar.warning("Please select your education level.")
    st.stop()
if Education == "Less than High School (Grades 1-8 or no formal schooling)":
    Education = 1
elif Education == "High School Incomplete (Grades 9-11 or Grade 12 with no diploma)":
    Education = 2
elif Education == "High School Graduate (Grade 12 with diploma or GED certificate)":
    Education = 3
elif Education == "Some college, no degree (includes some community college)":
    Education = 4
elif Education == "Two-year associate degree from a college or university":
    Education = 5
elif Education == "Four-year college or university degree/Bachelor's degree":
    Education = 6
elif Education == "Some postgraduate or professional schooling, no post-grad degree":
    Education = 7
else: Education = 8

Parent = st.sidebar.selectbox(
        "Are you a parent of a child under 18 living in your home?",
        ("Select response...","Yes", "No")
    )
## Parent Conversion
if Parent == "Select response...":
    st.sidebar.warning("Please select your parental status.")
    st.stop()
if Parent == "Yes":
    Parent = 1
else: Parent = 0

Married = st.sidebar.selectbox(
        "What is your current marital status?",
        ("Select response...","Married", "Living with a partner", "Divorced", "Separated", "Widowed", "Never been married")
    )

## Married Conversion
if Married == "Select response...":
    st.sidebar.warning("Please select your marital status.")
    st.stop()
if Married == "Married":
    Married = 1
else: Married = 0

Gender = st.sidebar.selectbox(
        "What is your gender identity?",
        ("Select response...","male", "female", "other")
    )

## Gender Conversion
if Gender == "Select response...":
    st.sidebar.warning("Please select your gender.")
    st.stop()
if Gender == "female":
    Gender = 1
else: Gender = 0

Age = st.sidebar.slider(label="Please enter your Age",
                min_value=1,
                max_value=97,
                value=35)

## Model Content
s = pd.read_csv("social_media_usage.csv")

## create df with the columns of interest from s
ss = s[['web1h','income','educ2','par','marital','gender','age']]
ss = ss.copy()

## creating other functions for replacing values
def clean_sm (x):
    output = np.where(x == 1,1,0)
    return output

def clean_income (x):
    output = np.where(x < 9, x, np.nan)
    return output

def clean_education (x):
    output = np.where(x < 8, x, np.nan)
    return output

def clean_age (x):
    output = np.where(x < 98, x,np.nan)
    return output

## replacing values

ss['sm_li'] = clean_sm(ss['web1h'])
ss['income'] = clean_income(ss['income'])
ss['educ2'] = clean_education(ss['educ2'])
ss['par'] = ss['par'].replace({1:1,2:0,8:np.nan,9:np.nan})
ss['marital'] = ss['marital'].replace({1:1,2:0,3:0,4:0,5:0,6:0,8:np.nan,9:np.nan})
ss['gender'] = ss['gender'].replace({1:0,2:1,3:0,8:np.nan,98:np.nan,99:np.nan})
ss['age'] = clean_age(ss['age'])

## Remove Missing values
ss = ss.dropna()

ss = ss.rename(columns={'par': 'parent',
                           'gender': 'female',
                           'marital': 'married',
                          'educ2':'education'})

## Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

## Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=12) # set for reproducibility

## Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')

## Fit algorithm to training data
lr.fit(X_train, y_train)

# New data from app user inputs: "income", "education", "parent", "married", "female", "age"

app_user = [Income, Education, Parent, Married, Gender, Age]

# Predict class, given input features
predicted_class = lr.predict([app_user])

# Generate probability of positive class (=1)
probs = lr.predict_proba([app_user])


# Main content - generated after inputs 
if st.session_state.button_clicked:
    st.write(f"The probability that {name} is a LinkedIn user is {probs[0][1]:.2%}.")

    if predicted_class == 1:
        prediction = "IS"
        st.image("images/linkedin_yes.png", caption="Looks like you're a LinkedIn user!", use_column_width=True)
    else:
        prediction = "IS NOT"
        st.image("images/linkedin_no.png", caption="Hmm…maybe LinkedIn isn't your thing.", use_column_width=True)
