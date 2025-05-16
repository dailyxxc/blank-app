import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import time
import logging  
from pathlib import Path
import joblib
import os
from pathlib import Path
import base64

# Set page title
st.markdown("<h1 style='text-align: center;'>PART1 Personalized Obesity Level Analysis Based on Eating Habits🥗</h1>", unsafe_allow_html=True)

# Sidebar success prompt
st.sidebar.success("Please complete data upload before using the sidebar functions❤")


# Get the absolute path of the script directory
path = str(Path(__file__).parent.absolute())

# Build the absolute path of the video file
video_file_path = Path(path) / "fit.mp4"

# Open the video file
video_file = open(video_file_path, 'rb')
video_bytes = video_file.read()

st.markdown(
    """
    ### 1. Significance
    With the change of lifestyle, the problem of obesity has become increasingly serious. Understanding and mastering the factors affecting obesity, and classifying and predicting the level of obesity, are of great significance for the formulation of obesity prevention and control strategies. This project aims to provide a basis for personalized obesity prevention and intervention through data analysis, helping people better manage their health.
"""
)


st.write("# Introduction🗠")
# Local video
st.video(video_bytes, format="mp4", start_time=2)
# Add source link
st.markdown("[Source Link](https://www.bilibili.com/video/BV1s54y1r7Ed/?spm_id_from=333.337.search-card.all.click&vd_source=096a17588c803a5145918c456427c022)")

# Main page content
st.markdown(
    """
    ### 2. Features
    Our dataset includes multiple features related to eating habits and physical conditions, such as gender, age, height, weight, family history of obesity, frequent consumption of high-calorie foods, daily vegetable intake, number of main meals per day, between-meal eating, smoking status, water intake, daily calorie monitoring status, physical activity frequency, time using electronic devices, alcohol consumption frequency, daily transportation mode, etc. These features have different degrees of correlation with the level of obesity. For example, family history of obesity and frequent consumption of high-calorie foods show a significant positive correlation with the level of obesity, while gender, smoking status, etc., have weaker correlations.
"""
)
import streamlit as st  
import pandas as pd  

# Create data
data = {  
    "Variable Name": [  
        "Gender",  
        "Age",  
        "Height",  
        "Weight",  
        "family_history_with_overweight",  
        "FAVC (Frequent High-Calorie Food Consumption)",  
        "FCVC (Daily Vegetable Intake)",  
        "NCP (Number of Main Meals Per Day)",  
        "CAEC (Between-Meal Eating)",  
        "SMOKE (Smoking Status)",  
        "Water Intake",  
        "CALC (Family Obesity History)",  
        "FAVC (Frequent High-Calorie Food Consumption)",  
        "NObeyesdad (Obesity Level)"  
    ],  
    "Role": [  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Feature",  
        "Target Variable"  
    ],  
    "Type": [  
        "Categorical Variable",  
        "Continuous Variable",  
        "Continuous Variable",  
        "Continuous Variable",  
        "Binary Variable",  
        "Binary Variable",  
        "Integer Variable",  
        "Continuous Variable",  
        "Categorical Variable",  
        "Binary Variable",  
        "Categorical Variable",  
        "Binary Variable",  
        "Binary Variable",  
        "Integer Variable"  
    ],  
    "Demographic Information": [  
        "Gender",  
        "Age",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None"  
    ],  
    "Description": [  
        "None",  
        "None",  
        "None",  
        "None",  
        "Do any family members have or have had overweight?",  
        "Do you frequently eat high-calorie foods?",  
        "Do you usually eat vegetables in your meals?",  
        "How many main meals do you have per day?",  
        "Do you eat any food between meals?",  
        "Do you smoke?",  
        "How much water do you drink per day?",  
        "Do any family members have or have had overweight?",  
        "Do you frequently eat high-calorie foods?",  
        "Obesity level classification"  
    ],  
    "Unit": [  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None"  
    ],  
    "Missing Values": [  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None",  
        "None"  
    ]  
}  

# Convert data to DataFrame
df = pd.DataFrame(data)  


# Display data table
st.dataframe(df)

st.markdown(
    """
    ### 3. Task
    The main task of this project is to use machine learning methods to accurately predict the level of obesity based on personal eating habits and physical condition data. We have built an effective prediction model through a series of steps such as data preprocessing, model selection, and optimization. Users can input relevant data and obtain personalized obesity level prediction results through interaction on the webpage, so as to better understand their health status and take corresponding prevention and intervention measures.
    """
)







# Global variables to store dataset and related models
df = None
encoder = None
columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
best_model = None

# User dataset upload function
def upload_dataset():
    global df, encoder, X, y, X_train, X_test, y_train, y_test, best_model
    st.subheader("→Please upload your dataset first!")
    uploaded_file = st.file_uploader("Please use your dataset file (.CSV format). If you don't have a specific dataset, you can download and use the test case dataset we prepared above.", type="csv")

    # Add model upload function
    uploaded_model_file = st.file_uploader("Please upload the model file (.pkl format). If you don't have a specific model, you can download and use the test model we prepared below.", type="pkl")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Data preprocessing
        encoder = LabelEncoder()
        for column in columns_to_encode:
            df[column] = encoder.fit_transform(df[column])

        # Split features and target variable
        X = df.drop('NObeyesdad', axis=1)
        y = df['NObeyesdad']

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if uploaded_model_file is not None:
            # Get the absolute path of the script directory
            path = str(Path(__file__).parent.absolute())

            # Build temporary model file storage path
            model_temp_path = Path(path) / "temp_model.pkl"

            # Save the uploaded model file to the temporary path
            with open(model_temp_path, 'wb') as f:
                f.write(uploaded_model_file.read())

            # Load the user-uploaded model
            best_model = joblib.load(model_temp_path)

            # Delete the temporary file
            os.remove(model_temp_path)

            st.sidebar.info(f"✅ Data uploaded! Your uploaded model has been successfully loaded! Please use the sidebar for personalized data configuration.")
        else:
            st.warning("Please upload a model file to proceed (use your trained model or our trained XGBoost model).")

    # Set xgb_model.pkl download button and absolute path
    # Get the absolute path of the script directory
    path = str(Path(__file__).parent.absolute())

    # Build the absolute path of xgb_model.pkl
    xgb_model_path = Path(path) / "xgb_model.pkl"

    if os.path.exists(xgb_model_path):
        with open(xgb_model_path, 'rb') as f:
            xgb_model_bytes = f.read()
        st.download_button(
            label="Download xgb_model.pkl (Test Model)",
            data=xgb_model_bytes,
            file_name="xgb_model.pkl",
            mime="application/octet-stream"
        )

def get_user_input_sidebar():
    st.sidebar.subheader("Please enter the following personal information:")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=1)
    gender_value = 1 if gender == "Male" else 0

    age = st.sidebar.number_input("Age", min_value=1, step=1)
    height = st.sidebar.number_input("Height (cm)", min_value=1, step=1)
    weight = st.sidebar.number_input("Weight (kg)", min_value=1, step=1)

    family_history = st.sidebar.selectbox("Family history of obesity", ["No", "Yes"], index=0)
    family_history_value = 1 if family_history == "Yes" else 0

    favc = st.sidebar.selectbox("Frequent consumption of high-calorie foods", ["No", "Yes"], index=0)
    favc_value = 1 if favc == "Yes" else 0

    fcvc = st.sidebar.selectbox("Vegetable intake in meals", ["No", "Yes"], index=1)
    fcvc_value = 1 if fcvc == "Yes" else 0

    ncp = st.sidebar.number_input("Number of main meals per day", min_value=1, step=1)

    caec = st.sidebar.selectbox("Eating between meals", ["No", "Yes"], index=0)
    caec_value = 1 if caec == "Yes" else 0

    smoke = st.sidebar.selectbox("Smoking status", ["No", "Yes"], index=0)
    smoke_value = 1 if smoke == "Yes" else 0

    ch2o = st.sidebar.number_input("Daily water intake (ml)", min_value=1, step=1)

    scc = st.sidebar.selectbox("Calorie monitoring", ["No", "Yes"], index=0)
    scc_value = 1 if scc == "Yes" else 0

    faf = st.sidebar.selectbox("Physical activity frequency", ["Rarely", "Occasionally", "Frequently"], index=1)
    faf_value = {"Rarely": 0, "Occasionally": 1, "Frequently": 2}[faf]

    tue = st.sidebar.number_input("Daily screen time (hours)", min_value=0, step=1)

    calc = st.sidebar.selectbox("Family members with overweight", ["No", "Yes"], index=0)
    calc_value = 1 if calc == "Yes" else 0

    mtrans = st.sidebar.selectbox("Transportation mode", ["Walking", "Cycling", "Driving", "Public Transport"], index=0)
    mtrans_value = {"Walking": 0, "Cycling": 1, "Driving": 2, "Public Transport": 3}[mtrans]

    user_data = {
        'Gender': gender_value,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history_value,
        'FAVC': favc_value,
        'FCVC': fcvc_value,
        'NCP': ncp,
        'CAEC': caec_value,
        'SMOKE': smoke_value,
        'CH2O': ch2o,
        'SCC': scc_value,
        'FAF': faf_value,
        'TUE': tue,
        'CALC': calc_value,
        'MTRANS': mtrans_value
    }

    if st.sidebar.button("Submit Information and Get Recommendations"):
        return pd.DataFrame(user_data, index=[0])
    else:
        return None

# Data validation function
def check_data(user_data):
    global df
    # Check if column names match training data
    expected_columns = df.columns.tolist()[:-1]  # Exclude target variable column
    if set(user_data.columns) != set(expected_columns):
        st.error("Input data column names do not match training data.")
        return False

    # Check categorical variable values are within valid range
    for column in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
        if user_data[column].iloc[0] not in [0, 1]:
            st.error(f"{column} value must be 0 or 1.")
            return False

    # Check FCVC value is within valid range (assumed 0 or 1)
    if user_data['FCVC'].iloc[0] not in [0, 1]:
        st.error("FCVC value must be 0 or 1.")
        return False

    # Check physical activity frequency (FAF) value is within valid range
    if user_data['FAF'].iloc[0] not in [0, 1, 2]:
        st.error("FAF value must be 0, 1, or 2.")
        return False

    # Check transportation mode (MTRANS) value is within valid range
    if user_data['MTRANS'].iloc[0] not in [0, 1, 2, 3]:
        st.error("MTRANS value must be 0, 1, 2, or 3.")
        return False

    return True

# Predict obesity level
def predict_obesity(user_data):
    global encoder
    user_data_encoded = user_data.copy()
    for column in columns_to_encode[:-1]:
        # Fit encoder first
        encoder.fit(df[column])
        user_data_encoded[column] = encoder.transform(user_data_encoded[column])
    prediction = best_model.predict(user_data_encoded)
    return prediction[0]

# Provide recommendations
def give_suggestions(prediction, user_data):
    if prediction == 0:
        st.success("Based on the information you provided, you may be underweight. Here are some targeted recommendations:")
        if user_data['Age'].iloc[0] < 18:
            st.write("You are in the growth and development stage. Ensure adequate intake of protein, calcium, and other nutrients, such as drinking more milk, eating eggs, and fish, to promote normal physical development.")
        else:
            st.write("It is recommended to increase nutrient intake, ensure regular three meals a day, appropriately increase foods rich in high-quality protein such as lean meat and beans, and intake of carbohydrates. You can choose healthy carbohydrate sources such as whole-grain bread and brown rice.")
    elif prediction == 1:
        st.success("Based on the information you provided, your weight is within the normal range. Continuing a healthy diet and appropriate exercise will help maintain good physical condition. Here are some maintenance recommendations:")
        if user_data['FAF'].iloc[0] == 0:
            st.write("Your current physical activity frequency is low. You can appropriately increase some simple exercises, such as walking for 30 minutes every day or practicing yoga twice a week, to further improve physical fitness.")
        elif user_data['FAF'].iloc[0] == 1:
            st.write("Your exercise frequency is acceptable. Continue to maintain your current exercise habits while paying attention to a balanced diet, eating more vegetables and fruits, and less greasy and sugary foods.")
        elif user_data['FAF'].iloc[0] == 2:
            st.write("You exercise frequently, which is great! Remember to arrange rest time reasonably, ensure sufficient sleep for the body to recover, and pay attention to the comprehensiveness of dietary nutrition.")
    elif prediction == 2:
        st.success("Based on the information you provided, you are in overweight class I. Here are some suggestions to help improve the situation:")
        if user_data['FAVC'].iloc[0] == 1:
            st.write("You frequently eat high-calorie foods, which may be one of the reasons for being overweight. It is recommended to control the intake of such foods, such as reducing the frequency of fried foods and desserts, and replacing them with vegetable salads and fruits.")
        if user_data['SMOKE'].iloc[0] == 1:
            st.write("Smoking has many adverse effects on health and may also be related to overweight. Consider quitting smoking or gradually reducing smoking, which will not only help control weight but also improve overall health.")
        st.write("Increase vegetable and fruit intake, and engage in at least three times a week of moderate-intensity exercise such as fast walking or jogging, with each exercise session recommended to be more than 30 minutes.")
    elif prediction == 3:
        st.success("Based on the information you provided, you are in overweight class II. It is necessary to strictly control diet more, reduce sugar and fat intake, increase exercise, and consider adding strength training to improve basal metabolic rate. Here are specific recommendations:")
        if user_data['CH2O'].iloc[0] < 1500:
            st.write("Your daily water intake may be insufficient. Sufficient water intake helps metabolism. It is recommended to drink at least 1500-2000 ml of water per day to help the body excrete waste and toxins.")
        if user_data['SCC'].iloc[0] == 0:
            st.write("You currently do not monitor daily calorie intake. It is recommended to start paying attention to calorie intake and use some diet recording apps to help you understand how many calories you eat each day, so as to better control your diet.")
        st.write("Increase exercise volume, and in addition to aerobic exercise, you can perform strength training 2-3 times a week, such as squats and planks, with each training session arranged for 20-30 minutes according to your own situation.")
    elif prediction == 4:
        st.success("Based on the information you provided, you belong to obesity type I. Here are some targeted recommendations:")
        if user_data['CALC'].iloc[0] == 1:
            st.write("There is a history of overweight or obesity in your family, and genetic factors may play a role in your weight problem. However, improvements can still be made through a healthy lifestyle. It is recommended to consult a professional dietitian to develop a personalized diet plan and increase exercise, such as at least 30 minutes of aerobic exercise every day, such as swimming or skipping rope.")
        if user_data['MTRANS'].iloc[0] == 2 or user_data['MTRANS'].iloc[0] == 3:
            st.write("Your transportation mode may be relatively sedentary, such as driving or taking public transportation. Try to increase opportunities for walking or cycling, such as getting off the bus one or two stops early and walking to the destination, or choosing to cycle for short trips to increase daily activity.")
        st.write("In terms of diet, strictly control the intake of high-calorie, high-fat, and high-sugar foods, and increase the intake of vegetables, fruits, and high-fiber foods.")
    elif prediction == 5:
        st.success("Based on the information you provided, you belong to obesity type II. It is strongly recommended that you seek the help of a doctor or professional health consultant as soon as possible to develop a comprehensive weight loss plan, including diet adjustment, exercise programs, and possibly medical interventions. Here are some preliminary recommendations:")
        if user_data['FCVC'].iloc[0] == 0:
            st.write("You usually do not eat vegetables in your meals, and vegetables are very important for maintaining health and controlling weight. Please be sure to increase vegetable intake, and ensure a certain amount of vegetables in each meal, which can be made into vegetable soup, stir-fried vegetables, and other forms.")
        if user_data['NCP'].iloc[0] > 3:
            st.write("You have more main meals per day, which may lead to excessive calorie intake. Consider appropriately reducing the number of main meals or controlling the amount of each meal while ensuring a balanced diet.")
        st.write("In terms of exercise, gradually increase the amount and intensity of exercise, and carry out systematic training under the guidance of professionals, including a combination of aerobic exercise and strength training.")
    elif prediction == 6:
        st.success("Based on the information you provided, you belong to obesity type III. This is a very serious case of obesity. Please seek medical attention immediately and make systematic treatment and lifestyle changes under the guidance of professionals. Here are some urgent recommendations:")
        if user_data['family_history_with_overweight'].iloc[0] == 1:
            st.write("You have a family history of obesity, and genetic factors plus your current weight status make the situation more severe. Please strictly follow the doctor's advice for treatment and lifestyle adjustments, including diet control, exercise arrangements, and possibly drug treatment.")
        if user_data['CAEC'].iloc[0] == 1:
            st.write("You eat between meals, which may further increase calorie intake. Please try to avoid eating between meals. If you need to eat, choose low-calorie, high-fiber foods such as fruits and nuts (in small amounts).")
        st.write("Comprehensive lifestyle changes are required, including strict diet management, regular exercise, and regular health check-ups.")

# Function to output all related indicators and their importance
def output_all_related_indicators(user_data):
    global best_model, X
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    importance_dict = dict(zip(feature_names, feature_importances))

    st.subheader("**3. Analysis of relevant indicators and their impact on weight status based on your information**")

    # Analyze the possible impact of gender on weight
    gender = "Male" if user_data['Gender'].iloc[0] == 1 else "Female"
    if user_data['Gender'].iloc[0] == 1:
        st.markdown(f"Your gender is <span style='color: lightblue; font-weight: bold;'>Male</span>👨. Generally, males may accumulate more muscle than females, and basal metabolic rate may be slightly higher, but this also depends on other lifestyle factors such as exercise and diet. Combining with other information you filled in, your weight status may be affected by multiple factors.", unsafe_allow_html=True)
    else:
        st.markdown(f"Your gender is <span style='color: lightpink; font-weight: bold;'>Female</span>👩. The body composition of females may have a higher proportion of fat compared to males, but a healthy weight can also be maintained through a reasonable diet and exercise. Based on the information you filled in, various factors are affecting your weight status.", unsafe_allow_html=True)

    # Analyze the possible impact of age on weight
    age = user_data['Age'].iloc[0]
    if age < 18:
        st.markdown(f"You are in the <span style='color: green; font-weight: bold;'>growing stage</span>🧒. The body is still in the growth and development stage, and weight changes at this stage may be more related to the needs of physical development. Reasonable nutrient intake is very important for healthy growth, and your current weight status is also affected by the characteristics of this stage and other living habits.", unsafe_allow_html=True)
    elif age < 30:
        st.markdown(f"You are in the <span style='color: orange; font-weight: bold;'>youth stage</span>👱. Metabolism is relatively fast, and it is generally easier to maintain weight, but if the diet is unbalanced or lacking in exercise, weight fluctuations may also occur. Combining the information you filled in, such as your diet and exercise habits, they are all affecting your current weight.", unsafe_allow_html=True)
    elif age < 50:
        st.markdown(f"You are in the <span style='color: purple; font-weight: bold;'>middle-aged stage</span>👨‍🦱. Metabolism may start to gradually slow down, and more attention needs to be paid to diet and exercise to maintain weight stability. From the information you provided, factors such as your diet structure (e.g., frequent consumption of high-calorie foods, vegetable intake in meals) and physical activity frequency are closely related to your current weight status.", unsafe_allow_html=True)
    else:
        st.markdown(f"You are in the <span style='color: gray; font-weight: bold;'>elderly stage</span>👴. Physical functions have declined, and metabolism is slower. Weight management may require more refined diet control and moderate exercise. The lifestyle information you filled in is affecting your current weight status.", unsafe_allow_html=True)

    # Analyze the relationship between height, weight, and their impact
    height = user_data['Height'].iloc[0]
    weight = user_data['Weight'].iloc[0]
    bmi = weight / ((height / 100) ** 2)
    st.markdown(f"Your BMI value calculated from the filled height ({height} cm) and weight ({weight} kg) is <span style='color: red; font-weight: bold;'>{bmi:.2f}</span>. Generally, different BMI ranges correspond to different weight status categories. You can refer to relevant standards to further understand your weight status. Combined with other filled information such as diet and exercise habits, they are also continuously affecting this value and your actual weight status.", unsafe_allow_html=True)

    # Analyze the possible impact of family obesity history on weight
    if user_data['family_history_with_overweight'].iloc[0] == 1:
        st.markdown(f"Your family has a history of obesity😟, which may mean you have a genetic predisposition to obesity. However, the risk of obesity can be effectively reduced through a healthy lifestyle such as a reasonable diet and regular exercise. Based on the information you filled in, these lifestyle habits are currently interacting with genetic factors to affect your weight.", unsafe_allow_html=True)
    else:
        st.markdown(f"Your family has no history of obesity🎉, which is an advantage to some extent. However, it is still necessary to maintain good living habits to maintain a healthy weight. From the information you filled in, your current diet, exercise, and other habits are all determining whether your weight can continue to be maintained within a healthy range.", unsafe_allow_html=True)

    # Analyze the impact of diet-related indicators on weight
    if user_data['FAVC'].iloc[0] == 1:
        st.markdown(f"You frequently eat high-calorie foods🍔, which is an important factor leading to weight gain. It is necessary to pay attention to controlling the intake of such foods. Combined with other diet-related information you filled in, such as vegetable intake in meals and calorie monitoring, they are all affecting your overall dietary calorie intake and thus your weight.", unsafe_allow_html=True)
    else:
        st.markdown(f"You do not frequently eat high-calorie foods🥗, which is helpful for maintaining weight. Together with other dietary information you filled in, such as vegetable intake in meals and the number of main meals per day, they jointly shape your current dietary calorie intake pattern and thus affect your weight.", unsafe_allow_html=True)

    if user_data['FCVC'].iloc[0] == 1:
        st.markdown(f"You usually eat vegetables in your meals🥦, which is a very good eating habit. Vegetables are rich in dietary fiber and other nutrients, which help increase satiety and promote intestinal peristalsis, playing a positive role in weight control. Combined with other diet-related information such as frequent consumption of high-calorie foods, they jointly affect your weight status.", unsafe_allow_html=True)
    else:
        st.markdown(f"You usually do not eat vegetables in your meals😕. Vegetables play an important role in the diet, and the lack of vegetable intake may lead to nutritional imbalance and insufficient dietary fiber, thereby affecting weight control. Combined with other diet-related information you filled in, such as the number of main meals per day, they are all affecting your weight.", unsafe_allow_html=True)

    ncp = user_data['NCP'].iloc[0]
    if ncp > 3:
        st.markdown(f"You have more main meals per day🍽️, which may lead to excessive calorie intake. It is necessary to pay attention to reasonably controlling the amount and type of food in each meal. Combined with other diet-related information you filled in, such as frequent consumption of high-calorie foods and calorie monitoring, comprehensively manage dietary calorie intake to affect weight.", unsafe_allow_html=True)
    else:
        st.markdown(f"The number of main meals you have per day is relatively reasonable👍. Together with other diet-related information you filled in, such as vegetable intake in meals, it helps maintain a healthy dietary calorie intake pattern and thus affects weight.", unsafe_allow_html=True)

    if user_data['CAEC'].iloc[0] == 1:
        st.markdown(f"You eat between meals🍪, which may increase extra calorie intake and is not conducive to weight control. Combined with other diet-related information you filled in, such as frequent consumption of high-calorie foods and calorie monitoring, you need to pay attention to reasonably choosing foods between meals to avoid weight gain.", unsafe_allow_html=True)
    else:
        st.markdown(f"You do not eat between meals🥳, which helps reduce unnecessary calorie intake and is helpful for maintaining weight. Combined with other diet-related information you filled in, such as vegetable intake in meals, it jointly shapes your current dietary calorie management and thus affects weight.", unsafe_allow_html=True)

    if user_data['SCC'].iloc[0] == 1:
        st.markdown(f"You monitor daily calorie intake📈, which is a very good weight management habit. It allows you to clearly understand your calorie intake. Combined with other diet-related information you filled in, such as frequent consumption of high-calorie foods and vegetable intake in meals, it helps you more accurately control your diet and thus affect your weight.", unsafe_allow_html=True)
    else:
        st.markdown(f"You do not monitor daily calorie intake😕, which may lead to unclear understanding of your calorie intake. Combined with other diet-related information you filled in, such as frequent consumption of high-calorie foods and vegetable intake in meals, it is recommended that you consider starting calorie monitoring to better manage your weight.", unsafe_allow_html=True)

    # Analyze the impact of exercise-related indicators on weight
    if user_data['SMOKE'].iloc[0] == 1:
        st.markdown(f"You smoke🚬. Smoking not only has many harmful effects on health but may also affect metabolism and thus weight. Combined with other information you filled in, such as physical activity frequency, it is recommended that you consider quitting smoking to improve overall health and weight management.", unsafe_allow_html=True)
    else:
        st.markdown(f"You do not smoke🎉, which is beneficial for physical health and weight management. Combined with other information you filled in, such as physical activity frequency, it helps maintain a healthy lifestyle and weight status.", unsafe_allow_html=True)

    if user_data['FAF'].iloc[0] == 0:
        st.markdown(f"Your physical activity frequency is very low😕, which is not conducive to maintaining weight and physical health. It is recommended to increase the frequency and intensity of physical activity. Combined with other information you filled in, such as daily screen time, reasonably arrange time for exercise to improve weight status.", unsafe_allow_html=True)
    elif user_data['FAF'].iloc[0] == 1:
        st.markdown(f"Your physical activity frequency is occasional👍, which is a passable situation, but you can further increase the frequency and intensity of physical activity. Combined with other information you filled in, such as daily screen time, better balance life and exercise to maintain a healthy weight status.", unsafe_allow_html=True)
    elif user_data['FAF'].iloc[0] == 2:
        st.markdown(f"Your physical activity frequency is relatively high👏, which is very good. Continue to maintain and reasonably arrange exercise intensity and rest time. Combined with other information you filled in, such as daily screen time, it helps maintain good weight status and physical health.", unsafe_allow_html=True)

    # Analyze the role of transportation mode on weight
    if user_data['MTRANS'].iloc[0] == 0:
        st.markdown(f"You choose walking🚶 as your transportation mode, which is a great way to increase daily activity and has a positive effect on weight control and physical health. Combined with other information you filled in, such as physical activity frequency, it helps maintain a healthy weight status.", unsafe_allow_html=True)
    elif user_data['MTRANS'].iloc[0] == 1:
        st.markdown(f"You choose cycling🚲 as your transportation mode, which is also a good way to increase daily activity and helps with weight control and physical health. Combined with other information you filled in, such as physical activity frequency, it helps maintain a healthy weight status.", unsafe_allow_html=True)
    elif user_data['MTRANS'].iloc[0] == 2:
        st.markdown(f"You choose driving🚗 as your transportation mode. Relatively speaking, driving involves less physical activity. It is necessary to pay attention to increasing physical activity in other aspects, such as increasing walking distance to and from the car or walking more after parking. Combined with other information you filled in, such as physical activity frequency, to maintain a healthy weight status.", unsafe_allow_html=True)
    elif user_data['MTRANS'].iloc[0] == 3:
        st.markdown(f"You choose public transport🚌 as your transportation mode. Although physical activity during the ride is limited, you can take advantage of opportunities such as getting on and off the bus to increase physical activity. Combined with other information you filled in, such as physical activity frequency, it helps maintain a healthy weight status.", unsafe_allow_html=True)

    # Display key information and analysis in a table
    st.markdown("### Summary of key information and analysis:")

    # Extract relevant information for table display
    gender = "Male" if user_data['Gender'].iloc[0] == 1 else "Female"
    age = user_data['Age'].iloc[0]
    bmi = user_data['Weight'].iloc[0] / ((user_data['Height'].iloc[0] / 100) ** 2)
    family_history = "Yes" if user_data['family_history_with_overweight'].iloc[0] == 1 else "No"
    high_calorie_food = "Yes" if user_data['FAVC'].iloc[0] == 1 else "No"
    vegetable_intake = "Yes" if user_data['FCVC'].iloc[0] == 1 else "No"
    meal_count = f"{user_data['NCP'].iloc[0]} meals"
    between_meal_eating = "Yes" if user_data['CAEC'].iloc[0] == 1 else "No"
    calorie_monitoring = "Yes" if user_data['SCC'].iloc[0] == 1 else "No"
    physical_activity_frequency = f"{user_data['FAF'].iloc[0]} (0=Rarely, 1=Occasionally, 2=Frequently)"
    transportation_mode = ["Walking", "Cycling", "Driving", "Public Transport"][user_data['MTRANS'].iloc[0]]

    # Create table data
    table_data = {
        "Indicator": ["Gender", "Age", "BMI", "Family Obesity History", "High-Calorie Food Intake", "Vegetable Intake", "Meal Count", "Between-Meal Eating", "Calorie Monitoring", "Physical Activity Frequency", "Transportation Mode"],
        "Status": [gender, f"{age} years old", f"{bmi:.2f}", family_history, high_calorie_food, vegetable_intake, meal_count, between_meal_eating, calorie_monitoring, physical_activity_frequency, transportation_mode],
        "Impact Analysis": [
            "Males may accumulate more muscle with slightly higher basal metabolism, affected by exercise and diet.",
            "Affects weight due to growth needs (under 18), fast metabolism (youth), slower metabolism (middle-aged/elderly).",
            f"BMI {bmi:.2f} indicates weight status, affected by diet and exercise.",
            "Genetic predisposition if yes; lifestyle determines health if no.",
            "Increases weight risk if yes; helps maintain weight if no.",
            "Aids weight control if yes; risks imbalance if no.",
            f"May cause excess calories if >3 meals; reasonable if ≤3 meals.",
            "May increase extra calories if yes; helps control intake if no.",
            "Helps manage weight if yes; unclear intake if no.",
            "Low frequency risks weight gain; high frequency aids health.",
            "Active modes (walking/cycling) help; sedentary modes (driving) require more exercise."
        ]
    }
    styled_table = pd.DataFrame(table_data)
    styled_table = styled_table.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('text-align', 'center'), ('font-size', '18px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '16px')]}
    ])
    # Set table height and width
    styled_table = styled_table.set_properties(**{'height': '500px', 'width': '100%'})

    # Display styled table
    st.dataframe(styled_table)
    # Provide health plan website link
    st.markdown("For more detailed health planning, you can refer to the following website: [Chinese Nutrition Society Official Website](https://www.cnsoc.org/)")

# New function: Show and provide test dataset download
def show_and_download_test_dataset():
    st.subheader("**Test Case Dataset**")
    # Get the absolute path of the script directory
    path = str(Path(__file__).parent.absolute())

    # Build the absolute path of the test data file
    test_file_path = Path(path) / "fixed_encoded_dataset.csv"

    # Read test data
    test_df = pd.read_csv(test_file_path)

    # Display test dataset
    st.dataframe(test_df)

    # Provide download button
    csv = test_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Test Case Dataset",
        data=csv,
        file_name="test_encoded_dataset.csv",
        mime="text/csv"
    )

# Main function
def main():
    st.markdown("<h1 style='text-align: center;'>PART2 Personalized Obesity Analysis Based on XGBoost and Feature Learning🥦</h1>", unsafe_allow_html=True)
    show_and_download_test_dataset()

    upload_dataset()

    user_data = get_user_input_sidebar()
    if user_data is not None:
        if check_data(user_data):
            prediction = predict_obesity(user_data)
            give_suggestions(prediction, user_data)
            output_all_related_indicators(user_data)
        else:
            st.error("Data validation failed. Please check your input data.")

if __name__ == "__main__":
    main()
