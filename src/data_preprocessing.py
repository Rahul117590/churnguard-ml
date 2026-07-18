import os
import logging
import pandas as pd


# make the file 
log_dirs='logs'
os.makedirs(log_dirs,exist_ok=True)

# set the logger
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# set the console_handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# set the file Handler
log_dirs=os.path.join(log_dirs,'data_preprocessing.log')
file_handler=logging.FileHandler(log_dirs)
file_handler.setLevel('DEBUG')

# formatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adder
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# load the data from the previous path
def load_data(train_path:str,test_path:str)->tuple:
    '''the data is load form the  previous data set'''
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        logger.info('the data is successfully loaded from the prvious path')
        return train_df,test_df
    except FileNotFoundError as e:
        logger.error(f'FileNotFoundError is generated due to {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error generated due to {e}')
        raise

def handle_duplicate_values(train_df:pd.DataFrame,test_df:pd.DataFrame)->tuple:
    ''' missing value is handed here'''
    try:
        
        train_df=train_df.drop_duplicates()
        test_df=test_df.drop_duplicates()
        logger.info('the data is successfully drop from the test_df and train_df')
        return train_df,test_df
    except ValueError as e:
        logger.error(f' Value erro is occure due to {e}')
        raise

    except Exception as e:
        logger.error(f' Unexpected error is due to {e}')
        raise

def handle_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    '''missing values handled here (TotalCharges blank spaces)'''
    try:
        train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
        test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')

        train_df['TotalCharges'] = train_df['TotalCharges'].fillna(train_df['TotalCharges'].median())
        test_df['TotalCharges'] = test_df['TotalCharges'].fillna(train_df['TotalCharges'].median())

        logger.info('missing values handled successfully')
        return train_df, test_df
    
    except Exception as e:
        logger.error(f'Unexpected error occurred due to {e}')
        raise


def encode_target(train_df:pd.DataFrame,test_df:pd.DataFrame):
    ''' data encoding is start from here '''
    try:
        train_df['Churn']=train_df['Churn'].map({'Yes':1,'No':0})
        test_df['Churn']=test_df['Churn'].map({'Yes':1,'No':0})
        logger.info('the data is successfully encoded here ')
        return train_df,test_df

    except ValueError as e:
        logger.error(f'the error is generated due  to {e}')
        raise
    except Exception as e:
        logger.error (f'Unexpected error is due to {e}')
        raise

def save_data(train_df:pd.DataFrame,test_df:pd.DataFrame,output_path:str):
    ''' the data is saved here from output path '''
    try:
        os.makedirs(output_path,exist_ok=True)
        train_df.to_csv(os.path.join(output_path,'train.csv'),index=False)
        test_df.to_csv(os.path.join(output_path,'test.csv'),index=False)
        logger.debug(f'the data is save successfully with the path of {output_path}')

    except FileNotFoundError as e:
        logger.error(f'the file is not exist here  due to {e}')
        raise
    except Exception as e:
        logger.error(f'the Unexpected error is generated due to {e}')
        raise
    
def main():
    ''' the data_preprocessing is stared here '''
    logger.info('===========data processing===========')
    try:
        
        train_path=os.path.join('data','raw','train.csv')
        test_path=os.path.join('data','raw','test.csv')
        train_df,test_df=load_data(train_path,test_path)
        train_df,test_df=handle_duplicate_values(train_df,test_df)
        train_df,test_df=handle_missing_values(train_df,test_df)
        train_df,test_df=encode_target(train_df,test_df)
        save_data(train_df,test_df,output_path=os.path.join('data', 'processed'))

    except FileNotFoundError as e:
        logger.error(f'this error is genreated due to {e}')
        raise
    except Exception as e:
        logger.error(f'this error is generated due to {e}')
        raise

if __name__=='__main__':
    main()
