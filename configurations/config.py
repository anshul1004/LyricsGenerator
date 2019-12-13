'''
Authors: Ashwani Kashyap, Ruchi Singh, Anshul Pardhi, Anant Srivastava
'''
class Config:
    def __init__(self):

        print("----- Initializing configuration")
        self.base_dataset_path = 'https://personal.utdallas.edu/~arp180012/'
        self.dataset_csv_file = 'songdata.csv'
        self.iterations = 10000000
        self.hidden_layer_size = 80
        self.seq_input_length = 30
        self.learning_rate = 0.1
        self.chars_to_predict = 200
        self.epsilon = 0.00000001
        self.print_iteration = 500
