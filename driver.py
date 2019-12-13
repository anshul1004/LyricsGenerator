'''
Authors: Ashwani Kashyap, Ruchi Singh, Anshul Pardhi, Anant Srivastava
'''
from utils.utils import Utils
from configurations.config import Config
from rnn.rnn import RnnModel

if __name__ == "__main__":

    # to configure the model with default values
    config = Config()

    # to initialise utilities used throughout the model
    utils = Utils()

    # read the data and pre-process it
    processed_data = utils.preprocess(config)

    # tokenize the data
    utils.tokenize(processed_data)

    # initialize the model
    rnn = RnnModel(utils, config)

    # train the model
    rnn.train()
