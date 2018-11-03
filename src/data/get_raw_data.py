from subprocess import call
import os
import logging

def main():
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('getting raw data')
    
    # set path of the raw data
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    
    # use Kaggle API to download the raw data
    logger.info('downloading train data')
    train_args = ['kaggle', 'competitions', 'download', 'titanic', '-f', 'train.csv', '-p', raw_data_path, '--force']
    call(train_args)
    logger.info('downloading train data completed')
    
    logger.info('downloading test data')
    test_args = ['kaggle', 'competitions', 'download', 'titanic', '-f', 'test.csv', '-p', raw_data_path, '--force']
    call(test_args)
    logger.info('downloading test data completed')

if __name__ == '__main__':
    # set up logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_fmt)
    
    # call the main method
    main()