import os
import logging
from xgboost import XGBClassifier
import pandas as pd
import joblib

# make the file 
log_dirs='logs'
os.makedirs(log_dirs,exist_ok=True)

# create the logger
logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

# create the streamhandler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# create the file handler
log_dirs=os.path.join(log_dirs,'model_training.log')
file_handler=logging.FileHandler(log_dirs)
file_handler.setLevel('DEBUG')

# create the formatter
formatter=logging.Formatter('%(asctime)s-%(levelname)s-%(name)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add the logger with the hadler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# load the data from the previous file 
def load_data(train_path:str):
    ''' the data is loaded from the prvious path '''
    try:
        train_df=pd.read_csv(train_path)
        #test_df=pd.read_csv(test_path)
        logger.info('the data is successfully is loaded ')
        return train_df
    except FileNotFoundError as e:
        logger.error(f' the file is not open due to {e}')
        raise
    except Exception as e:
        logger.error(f'this unexpected error is raise due to {e}')
        raise

def get_model(params:dict):
    ''' make the model of xgboost'''
    # make the object of the model
    try:
        model=XGBClassifier(**params)
        logger.info('the model is sucessfully crated here ')
        return model
    except ModuleNotFoundError as e:
        logger.error(f'this error happend due to {e}')
        raise
    except Exception as e:
        logger.error(f'the unexepected error happend due to {e}')
        raise
def train_model(model,train_df:pd.DataFrame):
    ''' the model traing is start form here '''
    try:
        # first divide the data into two form X_train and X_test
        X_train=train_df.drop(columns=['Churn'])
        y_train=train_df['Churn']
        logger.info('the data is divided into two form that is X_train,y_train')
        # now model is train the X_train and y_train
        models=model.fit(X_train,y_train)
        logger.info(f'the data is successfully trainned  with the model')
        return models
    except ModuleNotFoundError as e:
        logger.error(f'the model is not trained due to {e}')
        raise
    except Exception as e:
        logger.error(f'this unexected error is occured due to {e}')
        raise
def save_model(models,model_path:str):
    ''' the model is trained here '''
    try:
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        logger.info('the file is successfully created')
        joblib.dump(models,model_path)
        logger.info('the data model is sucessfully saved in the model folder')
    except FileNotFoundError as e:
        logger.error(f'the file in not save due to {e}')
        raise
    except Exception as e:
        logger.error(f'this unexpected error occur due to {e}')
        raise
def main():
    ''' the model_training is started here '''
    try:
        logger.info('=========model_taraining is started=========')
        train_path=os.path.join('data','final','train.csv')
        params={
                           'n_estimators':100,
                            'learning_rate':0.1,
                            'max_depth':4,
                            'min_child_weight':1,
                            'random_state':42
                        }
        train_df=load_data(train_path)
        model=get_model(params)
        models=train_model(model,train_df)
        save_model(models,model_path=os.path.join('models','model.pkl'))
        logger.info('the data is successfully saved in the models')
    except FileNotFoundError as e:
        logger.info(f'this error is happened with the {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error is happend due to {e}')
        raise



if __name__=='__main__':
    main()
    





