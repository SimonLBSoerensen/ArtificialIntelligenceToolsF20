import numpy as np

class GANNIndividual:
    """
    Genetic Algorithm: Individual (X genes, value encoding)
    """
    name = 'GA NN'

    input_size = 4 * 59 + 1  # 4 tokens per player & bias
    output_size = 4  # only 4 actions
    hidden_neurons = 100  # hidden neurons
    gene_count = hidden_neurons * input_size + hidden_neurons + hidden_neurons * output_size + output_size

    def __init__(self):
        self.hidden_neurons = 100
        self.input_size = 4 * 59 + 1
        self.output_size = 1

        self.W1 = np.zeros((self.hidden_neurons, self.input_size))
        self.W2 = np.zeros((self.output_size, self.hidden_neurons))

    def load_chromosome(self, chromosome):
        """
        Chromosome contains weights for NN
        """
        self.chromosome = chromosome
        w1_siz = self.hidden_neurons * self.input_size
        w2_siz = self.output_size * self.hidden_neurons
        self.W1 = self.chromosome[0: w1_siz].reshape((self.hidden_neurons, self.input_size))
        self.W2 = self.chromosome[w1_siz: w1_siz + w2_siz].reshape((self.output_size, self.hidden_neurons))

    # tanh
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    # softmax
    @staticmethod
    def softmax(x):
        return x

    def translate(self, state):
        '''
        Input is state to translate.
        Output is vector of size (59*4, 1)
        Where the first 59 numbers describes position of tokens for player 1
        The next 59 numbers describes position of tokens for player 2
        ...
        The last 59 numbers describes position of tokens for player 4
        '''
        repre = np.array([0] * (59 * 4)).reshape(4, 59)
        for i in range(4):
            for idx in state[i]:
                repre[i, min(idx, 58)] += 1
        return repre.ravel()

    def forward(self, state):

        temp_input = np.append(self.translate(state), 1)  # append '1' for bias
        net_input = np.ravel(temp_input).reshape((self.input_size, 1))

        activation = self.W1 @ net_input
        activation = activation * np.sqrt(1 / self.input_size)
        activation = self.tanh(activation)

        output = self.W2 @ activation
        # output = softmax(output)
        return output

    def evaluate_actions(self, state, next_states, dice_roll):
        action_values = np.zeros(4)
        for i, next_state in enumerate(next_states):
            if next_state is False:
                action_values[i] = -10000000
            else:
                action_values[i] = self.forward(next_state)
        return np.argmax(action_values)

    def play(self, state, dice_roll, next_states):
        """ Return action that evaluated to max value """
        return self.evaluate_actions(state, next_states, dice_roll)