import streamlit as st
import joblib
import pandas as pd

# 1. Load the "brain" you just trained
model = joblib.load('titanic_model.pkl')

import streamlit as st
import joblib
import pandas as pd

# 1. Load Model
model = joblib.load('titanic_model.pkl')

# --- Option 1 Visual Improvements ---
# A high-quality wide image for the header
st.image("https://images.pexels.com/photos/1018632/pexels-photo-1018632.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", caption="The Atlantic Ocean")

st.title("🚢 Titanic Survival Predictor")
st.markdown("---") # Visual divider line
st.write("Enter passenger details below to see if they would have survived.")
st.write("Enter passenger details below to see if they would have survived.")

# 2. Create the input fields (Sliders and Menus)
pclass = st.selectbox("Ticket Class (1=Luxury, 3=Economy)", [1, 2, 3])
sex = st.radio("Gender", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
parch = st.number_input("Parents/Children aboard", 0, 6, 0)
fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# 3. The Prediction Button
if st.button("Predict"):
    # Convert text to numbers for the model
    sex_val = 0 if sex == "male" else 1
    emb_map = {"S": 0, "C": 1, "Q": 2}
    emb_val = emb_map.get(embarked, 0)

    # Put data into a table the model understands
    input_df = pd.DataFrame(
        [[pclass, sex_val, age, sibsp, parch, fare, emb_val]],
        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    )

    # Run the prediction
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✨ This passenger likely **SURVIVED**.")
    else:
        st.error("💀 This passenger likely **DID NOT SURVIVE**.")