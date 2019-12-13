'''
Authors: Ashwani Kashyap, Ruchi Singh, Anshul Pardhi, Anant Srivastava
'''
import pandas


class Utils:
    def __init__(self):

        print("----- Initializing utilities")

        self.input_dimension = None
        self.chars_tokens = None
        self.tokens_chars = None
        self.data = None
        self.data_len = None

    def tokenize(self, processed_data):

        """
        Split processed data into total characters and unique characters
        :param processed_data: input lyrics to train
        :return: char to int mapping, int to char mapping, total unique characters
        """

        print("----- Tokenizing data-set")

        songs = processed_data
        unique_chars = list(set(songs))
        chars_tokens = dict()
        tokens_chars = dict()

        # tokenizing chars
        for i in range(len(unique_chars)):
            chars_tokens[unique_chars[i]] = i
            tokens_chars[i] = unique_chars[i]

        self.data = songs
        self.data_len = len(songs)
        self.chars_tokens = chars_tokens
        self.tokens_chars = tokens_chars
        self.input_dimension = len(unique_chars)

        print("----- Done tokenizing")
        print("-> input length: " + str(self.data_len))
        print("-> unique tokens: " + str(self.input_dimension))

        return self.chars_tokens, self.tokens_chars, self.input_dimension

    def preprocess(self, config):

        """
        Preprocess the input data
        :param config: input dataset and the hyperparameters
        :return: preprocessed data
        """

        print("----- Pre-processing data-set")

        # read csv file from location
        data = pandas.read_csv(config.base_dataset_path + config.dataset_csv_file)

        data_cols = self.print_artists(data)

        choice = input("Enter artist's id (Pressing any other key will lead in selection of all artists): ")

        # Select songs of the chosen artist from the data_set
        if choice.isdigit() and int(choice) <= len(data_cols) and int(choice) != 0:
            selected_artist = data_cols[int(choice) - 1]
            print("Selected artist = " + selected_artist)
            data = data.query('artist == @selected_artist')

        lyrics = data['text'].tolist()  # Extract lyrics of the selected artist from the dataset

        # Append the extracted lyrics in the preprocessed data to return
        result = ''
        for element in lyrics:
            result += element

        print("----- Done pre-processing the data-set")

        return result

    def print_artists(self, data):

        """
        Print all the artist names
        :param data: Input csv data
        :return: a list containing artist names
        """

        data_cols = data['artist'].unique().tolist()
        for index, value in enumerate(data_cols):
            print(index + 1, value)

        return data_cols
