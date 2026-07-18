import os
import logging
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
import matplotlib.pyplot as plt
import json

# for disply the model in the graph
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay

# make the file 
log_dirs='logs'
os.makedirs(log_dirs,exist_ok=True)

# create the logger
logger=logging.getLogger('model_evalulation')
logger.setLevel('DEBUG')

# create the console_hadler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# create the file 
log_dirs=os.path.join(log_dirs,'model_evaluation.log')
file_handler=logging.FileHandler(log_dirs)
file_handler.setLevel('DEBUG')

# create the formatter
formatter=logging.Formatter('%(asctime)s-%(levelname)s-%(name)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


# add logger to handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# load the data 
def load_model(model_path:str):
    '''the model data is loaded from here '''
    try:
        models=joblib.load(model_path)
        logger.info('the model is successfully loaded')
        return models
    except FileNotFoundError as e:
        logger.error(f'this error is gernerated dut to {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error is due to {e}')
        raise

def load_test_data(test_path:str)->tuple:
    ''' the test_data is loaded form here '''
    try:
        test_df=pd.read_csv(test_path)
        logger.info(f'the test data is sucessfully loaded')
        # we need to split the data for further use 
        X_test=test_df.drop(columns=['Churn'])
        y_test=test_df['Churn']
        logger.info('the data is successfully splited')

        return X_test,y_test
    except FileNotFoundError as e:
        logger.error(f'this error is dur to {e}')
        raise
    except Exception as e:
        logger.error(f'this Unexpected error is happed dur to {e}')

def model_evalution(model,X_test:pd.DataFrame,y_test:pd.DataFrame):
    '''model evulation is started from here '''
    try:
        ''' from there our data is start evaulating the model '''
        y_pred=model.predict(X_test)
        logger.info('the data us successfully predict the model')
        # now the time of find the accuracay of the model
        acc=accuracy_score(y_test,y_pred)
        print(f'the accurcay of the model is {acc:.4f}')
        rec=recall_score(y_test,y_pred)
        print(f'the recallsrcore of the model is {rec:.4f}')
        f1=f1_score(y_test,y_pred)
        print(f'the f1_socre of the model is {f1:.4f}')
        prec=precision_score(y_test,y_pred)
        print(f'the precision of metrix is {prec:.2f}')
        cm=confusion_matrix(y_test,y_pred)
        print(f'the value of confusion metrix is {cm}')
        logger.info(f'the model_evalution is sucessfully completed ')

        metrics={
            'accuracy of metrix':acc,
            'recall of models':rec,
            'f1_score of models':f1,
            'precison_score of model is ':prec,
            'confusion metrix of modele':cm.tolist(),

        }
        logger.info('the data of model_evaluation is store successfully ')
        return metrics
    except ValueError as e:
        logger.error(f'this error generated due to {e} ')
        raise
    except Exception as e:
        logger.error(f'this Unexpected error is gernerated due to {e}')
        raise
def plot(model,X_test,y_test,save_path:str):
    ''' here the model starting the graph between them '''
    try:
        # make the path for save figure
        os.makedirs(save_path,exist_ok=True)
        # figure of confusion metrics
        plt.figure(figsize=(10,6))
        ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)
        plt.title('confusion metrics')
        plt.savefig(os.path.join(save_path,'confusion_matric.png'))
        plt.close()

        # figure of roc curve 
        plt.figure(figsize=(10,6))
        RocCurveDisplay.from_estimator(model,X_test,y_test)
        plt.title('roc cureve metrics')
        plt.savefig(os.path.join(save_path,'RocCurve.png'))
        plt.close()

        # figure of precision call display
        plt.figure(figsize=(10,6))
        PrecisionRecallDisplay.from_estimator(model,X_test,y_test)
        plt.title('Precision recall ')
        plt.savefig(os.path.join(save_path,'precision_racall.png'))
        plt.close()
        logger.info('the model data is sucessfully ploted above ')

    except ValueError as e:
        logger.info(f'this error due to may be X_test is empty {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error is due to {e}')
        raise
def save_metrics(metrics:dict,output_path:str):
        ''' the metrics that store the data is sucessfully save in the ouput_path'''
        try:
            os.makedirs(output_path,exist_ok=True)
            metrics_path=os.path.join(output_path,'metrics.json')
            with open(metrics_path,'w') as f:
                json.dump(metrics,f,indent=4)
            logger.info(f'the metrics data is succesfullly store in the metrics.json foler')

        except FileNotFoundError as e:
            logger.error(f'this error is due to file not found in the path as raise error {e}')
            raise
        except Exception as e:
            logger.error(f'this Unexpected eerror is due to {e}')
            raise
def main():
        ''' the model_evaluation is start from here '''
        logger.info('========= model evaulation is started =========')
        try:
            model_path=os.path.join('models','model.pkl')
            test_path=os.path.join('data','final','test.csv')
            logger.info('the data is loaded successfully with there path')

            model=load_model(model_path)
            X_test,y_test=load_test_data(test_path)
            metrics=model_evalution(model,X_test,y_test)
            plot(model,X_test,y_test,save_path='artifacts/plot')
            save_metrics(metrics,output_path='reports')
            logger.info('the model_evalulation is sucessfully completed ')
        except FileNotFoundError as e:
            logger.error(f'this error is generated due to {e}')
            raise

        except Exception as e:
            logger.error(f'Unexpeceted error is generated due to {e}')
            raise

if __name__ =='__main__':
        main()


        







        
    

    

        