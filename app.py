import pandas as pd
from flask import Flask,render_template,request
import joblib

# make the object of flask 
app=Flask(__name__)

# load all the file from the data 
model=joblib.load('models/model.pkl')
scaler=joblib.load('models/scaler.pkl')
encoder=joblib.load('models/encoder.pkl')

# just divide the columne that which is catogorical and which are numerical
# categorical Col
categorical_cols=['gender','Partner','Dependents','PhoneService','MultipleLines',
                  'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                  'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
                  'PaymentMethod']
# Numerical Col
numeric_cols=['tenure','MonthlyCharges','TotalCharges']

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    # first need to take all the data from one dictionary with the order of index.html
    form_data={
        'gender':request.form['gender'],
        'SeniorCitizen':request.form['SeniorCitizen'],
        'Partner':request.form['Partner'],
        'Dependents':request.form['Dependents'],
        'tenure':float(request.form['tenure']),
        'PhoneService':request.form['PhoneService'],
        'MultipleLines':request.form['MultipleLines'],
        'InternetService':request.form['InternetService'],
        'OnlineSecurity':request.form['OnlineSecurity'],
        'OnlineBackup':request.form['OnlineBackup'],
        'DeviceProtection':request.form['DeviceProtection'],
        'TechSupport':request.form['TechSupport'],
        'StreamingTV':request.form['StreamingTV'],
        'StreamingMovies':request.form['StreamingMovies'],
        'Contract':request.form['Contract'],
        'PaperlessBilling':request.form['PaperlessBilling'],
        'PaymentMethod':request.form['PaymentMethod'],
        'MonthlyCharges':float(request.form['MonthlyCharges']),
        'TotalCharges':float(request.form['TotalCharges'])

    }
    # we need to convert this dictionary to DataFarme
    input_df=pd.DataFrame([form_data])
    '''
    convert the seinor_citzent to 0 and 1 because our actual data is in the 0 and 1 form and we take the
      input to user in yes or no so that why
    '''
    input_df['SeniorCitizen']=input_df['SeniorCitizen'].map({'Yes':1,'No':0})

    # encode the Categorial columne it means that the encode we did on the traning data and this is fresh input_data
    encode_array=encoder.transform(input_df[categorical_cols])
    '''take the column name of that data because when we apply ohe then it become dataFrame to array so that
         columne name is removed'''
    array_columns=encoder.get_feature_names_out(categorical_cols)

    # now convert that array to dataFrame
    encoded_df=pd.DataFrame(encode_array,columns=array_columns)

    # now scaled to numeric columns
    scaled_array=scaler.transform(input_df[numeric_cols])
    # convert this numeric columne to data Frame
    scaled_df=pd.DataFrame(scaled_array,columns=numeric_cols)

    # we also add the senior citigen with the output
    senior_df = input_df[['SeniorCitizen']].reset_index(drop=True)  
    #senior_df=pd.DataFrame[['SeniorCitizen']].reset_index(drop=True)
    # ready the final input
    # final_input=pd.concat([scaled_df,senior_df,encoded_df],axis=1)
    final_input = pd.concat([senior_df, scaled_df, encoded_df], axis=1)  # ✅ senior_df pehle


    # Now prediction 
    prediction=model.predict(final_input)[0]
    probability=model.predict_proba(final_input)[0][1]

    result='customer will churn ' if prediction == 1 else 'Customer will  stay'
    prob_percent =round(probability * 100, 3)
    return render_template('result.html',result=result,probability=prob_percent)


if __name__=='__main__':
    app.run(debug=True)