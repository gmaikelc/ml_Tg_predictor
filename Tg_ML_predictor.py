# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:41:37 2023

@author: Gerardo Casanola
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
#from molvs import Standardizer
from rdkit import Chem
#from openbabel import openbabel
from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
from rdkit.Chem import AllChem
import plotly.graph_objects as go

#Import Libraries
import pandas as pd
import numpy as np
import math 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# packages for streamlit
import streamlit as st
from PIL import Image
import io
import base64


#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Tg ML predictor', page_icon=":computer:", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped-header.png')
st.image(image)

st.write("[![Website](https://img.shields.io/badge/website-RasulevGroup-blue)](http://www.rasulev.org)")
st.subheader("📌" "About Us")
st.markdown("The group of Prof. Rasulev is focused on development of artificial intelligence (AI)-based predictive models to design novel polymeric materials, nanomaterials and to predict their various properties, including toxicity, solubility, fouling release properties, elasticity, degradation rate, biodegradation, etc. The group applies computational chemistry, machine learning and cheminformatics methods for modeling, data analysis and development of predictive structure-property relationship models to find structural factors responsible for activity of investigated materials.")


# Introduction
#---------------------------------#

st.title(':computer: _Tg ML predictor_ ')

st.write("""

**It is a free web-application for Glass Transition Temperature Prediction**

The glass transition temperature (Tg) is one of the most important properties of polymeric materials and indicates an approximate temperature below which a macromolecular system changes from a relatively soft, 
flexible and rubbery state to a hard, brittle and glass-like one1. The Tg value also determines the utilization limits of many rubbers and thermoplastic materials. 
Besides, the drastic changes in the mobility of the molecules in different glassy states (from the frozen to the thawed state) affect many other chemical and physical properties, 
such as mechanical modulus, acoustical properties, specific heat, viscosity, mechanical energy absorption, density, dielectric coefficients, viscosity and the gases and liquids difussion rate in the polymer material. 
The change of these mechanical properties also specifies the employment of the material and the manufacturing process.

The Tg ML predictor is a Web App that use a SVM regression model to predict the glass transition temperature. 

The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Mordred](https://github.com/mordred-descriptor/mordred), [MOLVS](https://molvs.readthedocs.io/), [Openbabel](https://github.com/openbabel/openbabel),
[Scikit-learn](https://scikit-learn.org/stable/)
**Workflow:**
""")


image = Image.open('workflow_TGapp.png')
st.image(image, caption='Tg ML predictor workflow')


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV file')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/gmaikelc/ml_Tg_predictor/main/example_file.csv) 
""")

uploaded_file_1 = st.sidebar.file_uploader("Upload a CSV file with AlvaDesc descriptors", type=["csv"])


#%% Reading data and reording if needed ####
####---------------------------------------------------------------------------####

def reading_reorder(data):
     
   
    #Select the specified columns from the DataFrame
    df_selected = data[loaded_desc]
    id = data.iloc[:,0]
    # Order the DataFrame by the specified list of columns
    test_data = df_selected.reindex(columns=loaded_desc)
    #descriptors_total = data[loaded_desc]
    
    return test_data, id


#%% normalizing data
### ----------------------- ###

def normalize_data(train_data, test_data):
    # Normalize the training data
    df_train = pd.DataFrame(train_data)
    saved_cols = df_train.columns
    min_max_scaler = preprocessing.MinMaxScaler().fit(df_train)
    np_train_scaled = min_max_scaler.transform(df_train)
    df_train_normalized = pd.DataFrame(np_train_scaled, columns=saved_cols)

    # Normalize the test data using the scaler fitted on training data
    np_test_scaled = min_max_scaler.transform(test_data)
    df_test_normalized = pd.DataFrame(np_test_scaled, columns=saved_cols)

    return df_train_normalized, df_test_normalized


#%% Determining Applicability Domain (AD)

def applicability_domain(x_test_normalized, x_train_normalized):
    
    X_train = x_train_normalized.values
    X_test = x_test_normalized.values
    # Calculate leverage and standard deviation for the training set
    hat_matrix_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    leverage_train = np.diagonal(hat_matrix_train)
    leverage_train=leverage_train.ravel()
    
    # Calculate leverage and standard deviation for the test set
    hat_matrix_test = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_test.T
    leverage_test = np.diagonal(hat_matrix_test)
    leverage_test=leverage_test.ravel()
    
    # threshold for the applicability domain
    
    h3 = 3*((x_train_normalized.shape[1]+1)/x_train_normalized.shape[0])  
    
    diagonal_compare = list(leverage_test)
    h_results =[]
    for valor in diagonal_compare:
        if valor < h3:
            h_results.append(True)
        else:
            h_results.append(False)         
    return h_results


#%% Removing molecules with na in any descriptor

#%% Removing molecules with na in any descriptor

def all_correct_model(test_data,loaded_desc, id_list):
    
    X_final = test_data[loaded_desc]
    X_final["ID"] = id_list
    # Assuming X_final is your DataFrame
    id_list = X_final["ID"]  # Extract the ID column
    X_final.drop(columns=["ID"], inplace=True)  # Drop the ID column from its original position
    X_final.insert(0, "ID", id_list)  # Insert the ID column at the first position
    
    rows_with_na = X_final[X_final.isna().any(axis=1)]         # Find rows with NaN values
    for molecule in rows_with_na["ID"]:
        st.write(f'\rMolecule {molecule} has been removed (NA value  in any of the necessary descriptors)')
    X_final1 = X_final.dropna(axis=0,how="any",inplace=False)
    
    id = X_final1["ID"]
    return X_final1, id

 # Function to assign colors based on confidence values
def get_color(confidence):
    """
    Assigns a color based on the confidence value.

    Args:
        confidence (float): The confidence value.

    Returns:
        str: The color in hexadecimal format (e.g., '#RRGGBB').
    """
    # Define your color logic here based on confidence
    if confidence == "HIGH" or confidence == "Inside AD":
        return 'lightgreen'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        return 'red'


#%% Predictions        

#%% Predictions        

def predictions(loaded_model, loaded_desc, df_test_normalized):
    scores = []
    h_values = []
    std_resd = []
    idx = X_final1['ID']

    descriptors_model = loaded_desc
        
    X = df_test_normalized[descriptors_model]
    predictions = loaded_model.predict(X)
    scores.append(predictions)
        
    # y_true and y_pred are the actual and predicted values, respectively
    
    # Create y_true array with all elements set to 2.56 and the same length as y_pred
    y_pred_test = predictions
    y_test = np.full_like(y_pred_test, mean_value)
    residuals_test = y_test -y_pred_test

    std_dev_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    std_residual_test = (y_test - y_pred_test) / std_dev_test
    std_residual_test = std_residual_test.ravel()
          
    std_resd.append(std_residual_test)
        
    h_results  = applicability_domain(X, df_train_normalized)
    h_values.append(h_results)
    

    dataframe_pred = pd.DataFrame(scores).T
    dataframe_pred.index = idx
    dataframe_pred.rename(columns={0: "Predictions"},inplace=True)
    
    dataframe_std = pd.DataFrame(std_resd).T
    dataframe_std.index = idx
    
        
    h_final = pd.DataFrame(h_values).T
    h_final.index = idx
    h_final.rename(columns={0: "Confidence"},inplace=True)
    

    std_ensemble = dataframe_std.iloc[:,0]
    # Create a mask using boolean indexing
    std_ad_calc = (std_ensemble >= 3) | (std_ensemble <= -3) 
    std_ad_calc = std_ad_calc.replace({True: 'Outside AD', False: 'Inside AD'})
   
    
    final_file = pd.concat([std_ad_calc,h_final,dataframe_pred], axis=1)
    
    final_file.rename(columns={0: "Std_residual"},inplace=True)
    
    h3 = 3*((df_train_normalized.shape[1]+1)/df_train_normalized.shape[0])  ##  Mas flexible

    final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Inside AD' )), 'Confidence'] = 'HIGH'
    final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Outside AD')), 'Confidence'] = 'LOW'
    final_file.loc[(final_file["Confidence"] == False) & ((final_file["Std_residual"] == 'Inside AD')), 'Confidence'] = 'MEDIUM'


            
    df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
    styled_df = df_no_duplicates.style.apply(lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],subset=["Confidence"], axis=1)
    
    return final_file, styled_df


#%% Create plot:

def final_plot(final_file):
    
    confident_tg = len(final_file[(final_file['Confidence'] == "HIGH")])
    medium_confident_tg = len(final_file[(final_file['Confidence'] == "MEDIUM")])
    non_confident_tg = len(final_file[final_file['Confidence'] == "LOW")])
    
    keys = ["High confidence", "Medium confidence", "Low confidence",]
    fig = go.Figure(go.Pie(labels=keys, values=[confident_tg, medium_confident_tg, non_confident_tg]))
    
    fig.update_layout(plot_bgcolor = 'rgb(256,256,256)', title_text="Global Emissions 1990-2011",
                            title_font = dict(size=25, family='Calibri', color='black'),
                            font =dict(size=20, family='Calibri'),
                            legend_title_font = dict(size=18, family='Calibri', color='black'),
                            legend_font = dict(size=15, family='Calibri', color='black'))
    
    fig.update_layout(title_text=None)
    
    return fig


#%%
def filedownload1(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Tg_ml_results.csv">Download CSV File with results</a>'
    return href

#%% RUN

data_train = pd.read_csv("data/" + "data_902_original_15desc_logTg_train.csv")
mean_value = data_train['logTg'].mean()


loaded_model = pickle.load(open("models/" + "svr_model.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "Tg_ml_descriptors.pickle", 'rb'))


if uploaded_file_1 is not None:
    run = st.button("RUN")
    if run == True:
        data = pd.read_csv(uploaded_file_1,) 
        train_data = data_train[loaded_desc]
        test_data, id_list =  reading_reorder(data)
        X_final1, id = all_correct_model(test_data,loaded_desc, id_list)
        X_final2= X_final1.iloc[:,1:]
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        final_file, styled_df = predictions(loaded_model, loaded_desc, df_test_normalized)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)

        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)
       

# Example file
else:
    st.info('👈🏼👈🏼👈🏼   Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example CSV Dataset with Alvadesc Descriptors'):
        data = pd.read_csv("example_file.csv")
        train_data = data_train[loaded_desc]
        test_data, id_list =  reading_reorder(data)
        X_final1, id = all_correct_model(test_data,loaded_desc, id_list)
        X_final2= X_final1.iloc[:,1:]
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        final_file, styled_df = predictions(loaded_model, loaded_desc, df_test_normalized)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)
        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
  
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; 
' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed by <a style='display: ;
 text-align: center' href="https://www.linkedin.com/in/gerardo-m-casanola-martin-27238553/" target="_blank">Gerardo M. Casanola</a> and  <a style='display:; 
  text-align: center' href="https://twitter.com/capigol" target="_blank">Lucas Alberca</a> for <a style='display: ; 
 text-align: center;' href="http://www.rasulev.org" target="_blank">RasulevGroup</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

