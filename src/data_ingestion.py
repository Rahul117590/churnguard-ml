import logging
import pandas as pd
import os

# now file make  by the oricess if make dir

log_dirs='log'
os.makedirs(log_dirs,'data_ingestion')

# imoprt logging
logger=logging.Logger('data_ingestion')
logger.setLevel('DBUG')

#StreamHandler
console_handler=logging.StreamHandler()
logger.setLevel('DEBUG')

#file handler
log_dirs=os.path.join(log_dirs,'data_ingestion.log')
file_handler=logging.FileHandler(log_dirs)
file_handler.setLevel('DEBUG')

# set formatter
formatter=logging.Formattter('%(asctime)s-%(setlevel)s-%(name)s-%(message)s')
console_handler.setFormattter(formatter)
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
        df=df.columns.str.strip()
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
        train_df=df.drop(columne=['Churn'])
        test_df=df['Churn']
        logger.info('the data split is successfully completed ')
        return train_df,test_df
    except ValueError as e:
        logger.error(f'Value Error is generated while {e}')

    except Exception as e:
        logger.error(f'Unexpected error is generated dur to {e}')
        raise
def save_data(train_df:pd.DataFrame,output_path:str)->None:
    '''data is save in by the spliting the data into train and test '''
    try:
        os.makedirs(os.path.join('data','raw','train.csv'))
        os.makedirs(os.path.join('data','raw','test.csv'))
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
        data_path=
        test_size=
        
        


