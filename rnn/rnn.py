'''
Authors: Ashwani Kashyap, Ruchi Singh, Anshul Pardhi, Anant Srivastava
'''
import numpy as np
import sys
# import random as rd


class RnnModel:
    def __init__(self, utils, config):

        print("----- Initializing RNN model")

        # initialize configurations and utilities of the model to be used throughout the training
        self.utils = utils
        self.config = config

        # initialize weight matrices with random numbers
        self.input_to_hidden_wt = np.random.randn(config.hidden_layer_size,
                                                  utils.input_dimension) * 0.01  # input to hidden
        self.hidden_to_hidden_wt = np.random.randn(config.hidden_layer_size,
                                                   config.hidden_layer_size) * 0.05  # input to hidden
        self.hidden_to_output_wt = np.random.randn(utils.input_dimension,
                                                   config.hidden_layer_size) * 0.02  # input to hidden
        self.hidden_layer_bias = np.zeros((config.hidden_layer_size, 1))
        self.output_layer_bias = np.zeros((utils.input_dimension, 1))

        # declare weight matrix states to be used throughout the training
        self.input_state = {}
        self.hidden_state = {}
        self.output_state = {}
        self.output_softmax_prob = {}

    def back_propagation(self, X, targets):

        """
        do a backward pass for the predicted outputs and updates the weights to minimise the loss
        :param X: input sequence
        :param targets: target output for the given input sequence
        :return: input to hidden gradient, hidden to hidden gradient, hidden to output gradient,
                    hidden layer bias gradient, output layer bias gradient, last hidden state
        """

        # initialize the weight matrices gradients
        input_to_hidden_gradient = np.zeros(shape=self.input_to_hidden_wt.shape)
        hidden_to_hidden_gradient = np.zeros(shape=self.hidden_to_hidden_wt.shape)
        hidden_to_output_gradient = np.zeros(shape=self.hidden_to_output_wt.shape)
        hidden_layer_bias_gradient = np.zeros(shape=self.hidden_layer_bias.shape)
        output_layer_bias_gradient = np.zeros(shape=self.output_layer_bias.shape)
        next_hidden_state_gradient = np.zeros(shape=self.hidden_state[0].shape)
        input_len = len(X)

        # update the weight gradients by computing derivative in reverse direction
        time_step = input_len-1
        while time_step >= 0:
            dy = np.copy(self.output_softmax_prob[time_step])

            dy[targets[time_step]] = dy[targets[time_step]] - 1
            hidden_state_transpose = self.hidden_state[time_step].transpose()
            hidden_to_output_gradient += np.dot(dy, hidden_state_transpose)
            output_layer_bias_gradient += dy

            hidden_to_output_wt_transpose = self.hidden_to_output_wt.transpose()
            hidden_layer_delta = np.dot(hidden_to_output_wt_transpose, dy) + next_hidden_state_gradient
            raw_hidden_layer = (1 - self.hidden_state[time_step] * self.hidden_state[time_step]) * hidden_layer_delta
            hidden_layer_bias_gradient = hidden_layer_bias_gradient + raw_hidden_layer

            input_state_t_transpose = self.input_state[time_step].transpose()
            input_to_hidden_gradient = input_to_hidden_gradient + np.dot(raw_hidden_layer, input_state_t_transpose)

            hidden_state_t_minus_transpose = self.hidden_state[time_step - 1].transpose()
            hidden_to_hidden_gradient = hidden_to_hidden_gradient + np.dot(raw_hidden_layer,
                                                                           hidden_state_t_minus_transpose)

            hidden_to_hidden_wt_transpose = self.hidden_to_hidden_wt.T.transpose()
            next_hidden_state_gradient = np.dot(hidden_to_hidden_wt_transpose, raw_hidden_layer)

            # wts_gradient_list = [input_to_hidden_gradient, hidden_to_hidden_gradient, hidden_to_output_gradient,
            #                      hidden_layer_bias_gradient, output_layer_bias_gradient]

            # for wt in wts_gradient_list:
            #     for i in range(len(wt)):
            #         for j in range(len(wt[0])):
            #             if wt[i][j] < -5:
            #                 wt[i][j] = -5
            #             elif wt[i][j] > 5:
            #                 wt[i][j] = 5

            # self.clip_weights(input_to_hidden_gradient, hidden_to_hidden_gradient,
            #                hidden_to_output_gradient, hidden_layer_bias_gradient, output_layer_bias_gradient)

            time_step -= 1

        return input_to_hidden_gradient, hidden_to_hidden_gradient, \
               hidden_to_output_gradient, hidden_layer_bias_gradient, \
               output_layer_bias_gradient, self.hidden_state[len(X) - 1]

    def forward_pass(self, X, Y, hidden_states):

        """
        do a forward pass, given a set of hidden state of weights and generates the predicted output
        :param X: input sequence
        :param Y: target output for the given input sequence
        :param hidden_states: hidden state of weights
        :return: cross entropy loss
        """

        # create a temporary hidden state (copy of prev hidden state)
        self.hidden_state[-1] = np.array(hidden_states, copy=True)

        input_len = len(X)

        # set cross entropy to zero
        cross_entropy_loss = 0
        idx = 0
        while idx < input_len:
            # map inputs with their one hot encoded values
            self.input_state[idx] = np.zeros((self.utils.input_dimension, 1))
            self.input_state[idx][X[idx]] = 1

            # compute thr dot product of input with their input to hidden wt matrix
            input_to_hidden_prod = np.dot(self.input_to_hidden_wt, self.input_state[idx])

            # compute thr dot product of hidden state with their hidden to hidden wt matrix
            hidden_to_hidden_prod = np.dot(self.hidden_to_hidden_wt, self.hidden_state[idx - 1])

            # compute the output of hidden state
            self.hidden_state[idx] = np.tanh(input_to_hidden_prod + hidden_to_hidden_prod + self.hidden_layer_bias)

            # compute the product of output wt hidden wt with their hidden to output wt matrix
            hidden_output_wt_hidden_prod = np.dot(self.hidden_to_output_wt, self.hidden_state[idx])

            # compute the un-normalized output
            self.output_state[idx] = hidden_output_wt_hidden_prod + self.output_layer_bias

            # apply soft-max to the output
            output_state_values_sum = np.sum(np.exp(self.output_state[idx]))
            self.output_softmax_prob[idx] = np.exp(self.output_state[idx]) / output_state_values_sum

            # compute the cross entropy loss of the predicted output
            cross_entropy_loss += -np.log(self.output_softmax_prob[idx][Y[idx]])
            idx += 1

        return cross_entropy_loss

    # def clip_weights(self, xh, hh, hy, hb, yb):
    #
    #     wts_gradient_list = [xh, hh, hy, hb, yb]
    #
    #     for wt in wts_gradient_list:
    #         for i in range(len(wt)):
    #             for j in range(len(wt[0])):
    #                 if wt[i][j] < -5:
    #                     wt[i][j] = -5
    #                 elif wt[i][j] > 5:
    #                     wt[i][j] = 5

    def train(self):

        """
        responsible for training of the model using the configurations and utilities assigned during initialization
        :return: void
        """

        print("----- Training the model")

        # assign memory wts for adagrad
        mem_input_to_hidden_wt = np.zeros(shape=self.input_to_hidden_wt.shape)
        mem_hidden_to_hidden_wt = np.zeros(shape=self.hidden_to_hidden_wt.shape)
        mem_hidden_to_output_wt = np.zeros(shape=self.hidden_to_output_wt.shape)
        mem_hidden_layer_bias = np.zeros(shape=self.hidden_layer_bias.shape)
        mem_output_layer_bias = np.zeros(shape=self.output_layer_bias.shape)

        # set the initial hidden state to None
        previous_hidden_state = np.zeros((self.config.hidden_layer_size, 1))

        curr_epoch_count = 0
        input_start_idx = 0

        # print generated text with random wts
        generated_txt = self.generate_sequence(np.zeros((self.config.hidden_layer_size, 1)),
                                               self.utils.chars_tokens[self.utils.data[0]],
                                               self.config.chars_to_predict)

        print("----- Initial Generated text with random weights")
        print("---------------------")
        print(generated_txt)
        print("---------------------")

        # do multiple forward and backward passes until done with the iterations
        for curr_epoch_count in range(self.config.iterations):

            # if the whole input is traversed mark the completion of epoch
            previous_hidden_state, input_start_idx = self.check_for_input_overflow(previous_hidden_state,
                                                                                   input_start_idx)

            # generate the sequence based on previous hidden state and updated wts
            if curr_epoch_count != 0 and (curr_epoch_count % self.config.print_iteration == 0 \
                    or curr_epoch_count == 1 or (curr_epoch_count <= 500 and curr_epoch_count % 50 == 0)) \
                    and previous_hidden_state is not None:

                # To print in console stdout -
                print('-> Iteration: ' + str(curr_epoch_count) + " with cross_entropy_loss: " + str(cross_entropy_loss))
                print('---------------------')
                generated_txt = self.generate_sequence(previous_hidden_state, X[0], self.config.chars_to_predict)
                print(generated_txt)
                print("---------------------")

                # # To write a console output in file -
                # sys.stdout = open('console_output.txt', 'a+')
                #
                # print('-> Iteration: ' + str(curr_epoch_count) + " with cross_entropy_loss: "
                #       + str(cross_entropy_loss))
                # print('---------------------')
                # generated_txt = self.generate_sequence(previous_hidden_state, X[0], self.config.chars_to_predict)
                # print(generated_txt)
                # print("---------------------")

            # fetch new chunk of inputs and target outputs
            X, Y = self.get_input_target_tokens(input_start_idx)

            # run a forward pass
            cross_entropy_loss = self.forward_pass(X, Y, previous_hidden_state)

            # run a backward pass
            input_to_hidden_wt_change, hidden_to_hidden_wt_change, hidden_to_output_wt_change, hidden_bias_change, \
            output_bias_change, previous_hidden_state = self.back_propagation(X, Y)

            # update wts with the gradients using concept of adagrad
            self.input_to_hidden_wt, mem_input_to_hidden_wt = self.adagrad(self.input_to_hidden_wt,
                                                                           input_to_hidden_wt_change,
                                                                           mem_input_to_hidden_wt)

            self.hidden_to_hidden_wt, mem_hidden_to_hidden_wt = self.adagrad(self.hidden_to_hidden_wt,
                                                                             hidden_to_hidden_wt_change,
                                                                             mem_hidden_to_hidden_wt)
            self.hidden_to_output_wt, mem_hidden_to_output_wt = self.adagrad(self.hidden_to_output_wt,
                                                                             hidden_to_output_wt_change,
                                                                             mem_hidden_to_output_wt)
            self.hidden_layer_bias, mem_hidden_layer_bias = self.adagrad(self.hidden_layer_bias, hidden_bias_change,
                                                                         mem_hidden_layer_bias)
            self.output_layer_bias, mem_output_layer_bias = self.adagrad(self.output_layer_bias, output_bias_change,
                                                                         mem_output_layer_bias)

            # increase the index for the next chunk
            input_start_idx += self.config.seq_input_length

    def check_for_input_overflow(self, previous_hidden_state, start_idx):

        """
        checks if the next input chunk is overflowing
        :param previous_hidden_state: previous hidden state
        :param start_idx: next input chunk start index
        :return: updated hidden state, updated next input chunk start index
        """

        if start_idx + self.config.seq_input_length + 1 >= self.utils.data_len:
            # re-initialize the previous hidden state, and input start index
            previous_hidden_state = np.zeros((self.config.hidden_layer_size, 1))
            start_idx = 0

        return previous_hidden_state, start_idx

    def get_input_target_tokens(self, input_start_idx):

        """
        generates tokenized inputs and their target outputs
        :param input_start_idx: starting index for the input to be taken from the data_set
        :return: tokenized input, tokenized target outputs
        """

        X, Y = [], []
        # set target output y as the input at timestamp t + 1, for input x at timestamp t
        for i in range(input_start_idx, input_start_idx + self.config.seq_input_length):
            X.append(self.utils.chars_tokens[self.utils.data[i]])
            Y.append(self.utils.chars_tokens[self.utils.data[i + 1]])
        return X, Y

    def generate_sequence(self, hidden_state, input_char, char_to_generate):

        """
        generates the predicted sequence
        :param hidden_state: hidden state of weight matrix
        :param input_char: initial input character token
        :param char_to_generate: number of characters to be generated
        :return: generated sequence
        """

        # set the one hot encoding of the input
        input = np.zeros((self.utils.input_dimension, 1))
        input[input_char] = 1

        # initialize generated sting as empty
        output_string = ""

        char_count = 0
        while char_count < char_to_generate:

            # compute thr dot product of input with their input to hidden wt matrix
            input_to_hidden_prod = np.dot(self.input_to_hidden_wt, input)

            # compute thr dot product of hidden state with their hidden to hidden wt matrix
            hidden_to_hidden_prod = np.dot(self.hidden_to_hidden_wt, hidden_state)

            # compute the output of hidden state
            hidden_state = np.tanh(input_to_hidden_prod + hidden_to_hidden_prod + self.hidden_layer_bias)

            # compute the product of output wt hidden wt with their hidden to output wt matrix
            hidden_output_wt_hidden_prod = np.dot(self.hidden_to_output_wt, hidden_state)

            # compute the un-normalized output
            output = hidden_output_wt_hidden_prod + self.output_layer_bias

            output_sum = np.sum(np.exp(output))
            # apply soft-max to the output
            normalized_output = np.exp(output) / output_sum

            # ix = self.pick_max_prob_idx(normalized_output)
            # print("my method prob idx: " + str(ix) + " with value" + str(normalized_output[ix]) + " with char "
            #       + str(self.utils.tokens_chars[ix]) )

            # pick the one randomly based on weighted probability distribution
            idx = np.random.choice(range(self.utils.input_dimension), p=normalized_output.ravel())

            # append the predicted char
            output_string = output_string + self.utils.tokens_chars[idx]

            # encode next char with its corresponding one hot encoded value
            input = np.zeros((self.utils.input_dimension, 1))
            input[idx] = 1
            char_count += 1

        return output_string

    # def pick_max_prob_idx(self, output):
    #
    #     max_prob = float("-inf")
    #
    #     max_idx = -1
    #     curr_idx = 0
    #     for x in np.nditer(output):
    #         if x > max_prob or (max_prob == x and rd.randint(0, 1) == 1):
    #             max_prob = x
    #             max_idx = curr_idx
    #         curr_idx += 1
    #
    #     return max_idx

    def adagrad(self, weights, derivative_weights, mem_weights):

        """
        to reduce the learning weights with each iteration (adaptive learning rate)
        :param weights: weight matrix
        :param derivative_weights: weight matrix gradient
        :param mem_weights: previous weight matrix
        :return: updated weights, updated previous weights
        """

        square_weights = derivative_weights * derivative_weights
        mem_weights += square_weights
        delta_weight = self.config.learning_rate * derivative_weights / np.sqrt(mem_weights + self.config.epsilon)
        weights -= delta_weight
        return weights, mem_weights
