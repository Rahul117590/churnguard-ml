import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# now file make  by the oricess if make dir

log_dirs='logs'
os.makedirs(log_dirs,exist_ok=True)

# imoprt logging
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#StreamHandler
console_handler=logging.StreamHandler()
logger.setLevel('DEBUG')

#file handler
log_dirs=os.path.join(log_dirs,'data_ingestion.log')
file_handler=logging.FileHandler(log_dirs)
file_handler.setLevel('DEBUG')

# set formatter
formatter=logging.Formatter('%(asctime)s-%(levelname)s-%(name)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#adder to handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

#data ingestion is start form here 
def data_load(data_path:str)->pd.DataFrame:
    ''' data is loaded form here'''
    try:
        df=pd.read_csv(data_path)
        logger.debug(f'the is processes with the data_path {data_path} with the shape{df.shape} ')
        return df
    except FileNotFoundError as e:
        logger.error(f'the error is may be due to data is not found in ther is loaction {e}')
        raise
    except Exception as e:
        logger.error(f'the error is raise due to {e}')
        raise

def drop_unwanted_columns(df:pd.DataFrame)->pd.DataFrame:
    ''' the unwanted columne is drop from here '''
    try:
        df.columns=df.columns.str.strip()
        df=df.drop(columns=['customerID'])
        logger.info('the data drop all the unwanted columne name')
        return df
    except AttributeError as e:
        logger.error('AttributeError occure while removing the unwated columne {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error raise due to {e} ')
        raise

def split_data(df:pd.DataFrame,test_size:float)->tuple:
    '''now data split is started form here '''
    try:
        logger.info('the data split is successfully completed ')
        train_df,test_df=train_test_split(df,test_size=test_size,random_state=42)
        return train_df,test_df
    except ValueError as e:
        logger.error(f'Value Error is generated while {e}')

    except Exception as e:
        logger.error(f'Unexpected error is generated dur to {e}')
        raise
def save_data(train_df:pd.DataFrame,test_df:pd.DataFrame,output_path:str):
    '''data is save in by the spliting the data into train and test '''
    try:
        os.makedirs(output_path,exist_ok=True)
        train_df.to_csv(os.path.join(output_path,'train.csv'),index=False)
        test_df.to_csv(os.path.join(output_path,'test.csv'),index=False)
        logger.debug('the data is save in the path of train.csv and test.csv')
    except FileNotFoundError as e:
        logger.error(f'the data is not saved properly due to {e}')
        raise
    except Exception as e:
        logger.error(f'the Unexpected error raise due to {e}')
        raise
def main():
    '''data ingestin is start from here '''
    logger.info('======== data_ingestin is start')
    try:
        data_path="https://raw.githubusercontent.com/Rahul117590/Data-for-spam_prediction/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        test_size=0.2
        df=data_load(data_path=data_path)
        logger.info('data_load is completed')
        df=drop_unwanted_columns(df)
        logger.info('unwanted columns is droped')
        train_df,test_df=split_data(df,test_size=test_size)
        logger.info('the data split is completed')
        save_data(train_df,test_df,output_path=os.path.join('data','raw'))
        logger.info('the data save is completed')
    except FileNotFoundError as e:
        logger.error(f'the FileNotFound is error is due to {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error is due to {e}')
        raise


        


if __name__=='__main__':
    main()