# Import libraries
import streamlit as st
import pandas as pd
import pickle

 # Header
st.title('Fetal Health Classification: A Machine Learning App') 
st.image('fetal_health_image.gif', width = 400)
st.subheader("Utilize our advanced Machine Learning application to predict fetal health classfications.") 
st.write("To ensure optimal results, please ensure that your data strictly adheres to the specified format outlined below:") 

provided_data = pd.read_csv('fetal_health.csv')
first5 = provided_data.head(5)
st.dataframe(first5)

# Upload
uploaded_data = st.file_uploader('Upload your data')

if uploaded_data is not None:
  user_df = pd.read_csv(uploaded_data) # User provided data

  # Dropping null values
  user_df = user_df.dropna() 
  provided_data = provided_data.dropna()

  # Remove output column from original data
  provided_data = provided_data.drop(columns = "fetal_health")

  # Ensure the order of columns in user data is in the same order as that of original data
  user_df = user_df[provided_data.columns]
  original_rows = provided_data.shape[0]

  rf_pickle = open('rf_fetal_health.pickle', 'rb') 
  rf_model = pickle.load(rf_pickle) 
  rf_pickle.close() 
  
  # Predictions for user data
  user_pred = rf_model.predict(user_df)
  user_pred_prob = rf_model.predict_proba(user_df)
  user_df['Predicted Fetal Health'] = user_pred
  user_df['Prediction Probability (%)'] = user_pred_prob.max(axis = 1)


  # Show the predicted species on the app
  st.subheader("Predicting Fetal Health Class") 

  # Mapping results
  value_mapping = {0 : 'Suspect', 1 : 'Normal', 2 : 'Pathological'}
  user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].replace(value_mapping)

  # Styling the dataframe
  def style_cells(val):
    if val == 'Suspect':
        color = 'yellow'
    elif val == 'Normal':
        color = 'green'
    else:
        color = 'orange'
    
    return f'background-color: {color}'
  
  # Apply the style function to the DataFrame
  styled_user_df = user_df.style.applymap(style_cells, subset=['Predicted Fetal Health'])
  st.dataframe(styled_user_df)

  # Visualizing Results
  st.subheader("Prediction Performance") 

  tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
  with tab1:
    st.image('feature_imp.svg')
  with tab2:
    st.image('confusion_mat.svg')
  with tab3:
    class_report_df = pd.read_csv('class_report.csv')
    st.dataframe(class_report_df)

else: 
  st.write("Please upload a CSV file.")

# Create dummies for the combined dataframe
  # combined_df_encoded = pd.get_dummies(combined_df)

  # Split data into original and user dataframes using row index
  # original_df_encoded = combined_df_encoded[:original_rows]
  # user_df_encoded = combined_df_encoded[original_rows:]



