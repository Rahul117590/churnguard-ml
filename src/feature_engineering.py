import os
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.preprocessing import StandardScaler


#make the file 
log_dirs='logs'
os.makedirs(log_dirs,exist_ok=True)

# create the logger
logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# create the consoleHandler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# create the fileHandler
log_dirs=os.path.join(log_dirs,'featuring_engineering.log')
file_handler=logging.FileHandler(log_dirs)
file_handler.setLevel('DEBUG')

# create the formatter
formatter=logging.Formatter('%(asctime)s-%(levelname)s-%(name)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add the addhandler
logger.addHandler(console_handler)
logger.addHandler(file_handler)



# now the first data load is started form here 
def data_load(train_path:str,test_path:str)->tuple:
    ''' the data load is started from here '''
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        logger.info(f'the data loaded is succesfully created')
        return train_df,test_df

    except FileNotFoundError as e:
        logger.error(f'thisi eroor is created may be due to {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error due to {e}')
        raise
def encoder_categorical(train_df:pd.DataFrame,test_df:pd.DataFrame)->tuple:
    ''' the data encoder is created here  '''
    try:
        #ready the columne on which OHE used to apply
        catgorical_col=(['gender','Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'])
        # these are the column on which onehot encoder used to apply 
        

        # make the object of the encoder 
        encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

        #now train the encoder with the train data other wise 
        encoder.fit(train_df[catgorical_col])

        # now we tranform the data 
        train_encode_array=encoder.transform(train_df[catgorical_col])
        test_encode_array=encoder.transform(test_df[catgorical_col])
        logger.info('the encoded array is sucessfully here ')
        

        # now we need to convert the data into dataFrame
        # copy the column name of the data
        encoded_column_name=encoder.get_feature_names_out(catgorical_col)
        #now convert the data into the dataFrame
        train_df_encoded=pd.DataFrame(train_encode_array,columns=encoded_column_name)
        test_df_encoded=pd.DataFrame(test_encode_array,columns=encoded_column_name)
        logger.info('the data is successfully changed into the data frame')

        # now remove the columne of old categories name
        train_df=train_df.drop(columns=catgorical_col).reset_index(drop=True)
        test_df=test_df.drop(columns=catgorical_col).reset_index(drop=True)
        logger.info('the data remove the old categorical columns')

        # now add the train_df,test_df with train_encoded array and test_encoded array
        train_df=pd.concat([train_df,train_df_encoded],axis=1)
        test_df=pd.concat([test_df,test_df_encoded],axis=1)
        logger.info('the train_df and test_df is sucessfully converted into data frame')

        # save the encoded file 
        os.makedirs('models',exist_ok=True)
        joblib.dump(encoder,os.path.join('models','encoder.pkl'))
        logger.info('the model is sucessfully saved in the model.pkl')
        return train_df,test_df

    except TypeError as e:
        logger.error(f'the file do not saved due to {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error is due to {e}')
        raise

def scale_feature(train_df:pd.DataFrame,test_df:pd.DataFrame)->tuple:
    ''' the standardazation is going to sart form here '''
    try:
        scaler=StandardScaler()
        #now need to apply the scaler feature to all the feature of the data
        numeric=['tenure','MonthlyCharges','TotalCharges'] 
        # now we fit the X_train model of the model
        scaler.fit(train_df[numeric])
        logger.info('the X_train is successfully fited here')
        # now we trainsfrom the both X_train and X_test_model
        train_df[numeric]=scaler.transform(train_df[numeric])
        test_df[numeric]=scaler.transform(test_df[numeric])
        logger.info('the X_train and X_test is successfully compeleted here ')

        # we have to save this model data 
        # first we create the folder of scaler
        os.makedirs('models',exist_ok=True)
        # now we have to create the file with that 
        joblib.dump(scaler,os.path.join('models','scaler.pkl'))
        logger.debug(f'the data is sucessfully scaled and stored in the folder of ')
        return train_df,test_df
    except FileNotFoundError as e:
        logger.error(f'the file is not found here due to {e}')
        raise
    except Exception as e:
        logger.error(f'the Unexpected error is happend due to {e}')
        raise

def save_data(train_df:pd.DataFrame,test_df:pd.DataFrame,output_path:str):
    ''' the data save is started form here '''
    try:
        os.makedirs(output_path,exist_ok=True)
        train_df.to_csv(os.path.join(output_path,'train.csv'),index=False)
        test_df.to_csv(os.path.join(output_path,'test.csv'),index=False)
        logger.debug(f'the data is store in the path of {output_path}')
    except FileExistsError as e:
        logger.error(f'this error is generated due to {e}')
        raise
    except Exception as e:
        logger.info(f'Unexpected error happend due to {e}')
        raise
def main():
    ''' feature engineering is stared here'''
    logger.info('==========feature engineering is start==========')
    try:
        train_path=os.path.join('data','processed','train.csv')
        test_path=os.path.join('data','processed','test.csv')

        train_df,test_df=data_load(train_path,test_path)
        train_df,test_df=encoder_categorical(train_df,test_df)
        train_df,test_df=scale_feature(train_df,test_df)

        save_data(train_df,test_df,output_path=os.path.join('data','final'))
        logger.info('========feature engineering is completed sucessfully========')

    except ValueError as e:
        logger.error(f'the data is invalid due to {e}')
        raise
    except Exception as e:
        logger.error(f'the Unexpected error is generated due to {e}')
        raise

if __name__=='__main__':
    main()
    







        





