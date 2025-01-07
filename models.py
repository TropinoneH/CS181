import numpy as np

import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            terminated = True
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) != self.get_prediction(x):
                    self.w.update(x, nn.as_scalar(y))
                    terminated = False
            if terminated:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(1, 32)
        self.b1 = nn.Parameter(1, 32)
        self.w2 = nn.Parameter(32, 32)
        self.b2 = nn.Parameter(1, 32)
        self.w3 = nn.Parameter(32, 1)
        self.b3 = nn.Parameter(1, 1)
        self.alpha = 1e-2

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.w2), self.b2))
        x = nn.AddBias(nn.Linear(x, self.w3), self.b3)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        allLoss = []
        epoch = 0
        while True:
            lossList = []
            # print(f'Epoch {epoch}:')
            for x, y in dataset.iterate_once(batch_size=100):
                loss = self.get_loss(x, y)
                lossList.append(nn.as_scalar(loss))
                grad_m1, grad_b1, grad_m2, grad_b2, grad_m3, grad_b3 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(grad_m1, -self.alpha)
                self.b1.update(grad_b1, -self.alpha)
                self.w2.update(grad_m2, -self.alpha)
                self.b2.update(grad_b2, -self.alpha)
                self.w3.update(grad_m3, -self.alpha)
                self.b3.update(grad_b3, -self.alpha)
            lossSum = sum(lossList)
            allLoss.append(lossSum)
            if len(allLoss) > 2 and allLoss[-1] > allLoss[-2]:
                break
            epoch += 1


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.m1 = nn.Parameter(784, 1024)
        self.b1 = nn.Parameter(1, 1024)
        self.m2 = nn.Parameter(1024, 240)
        self.b2 = nn.Parameter(1, 240)
        self.m3 = nn.Parameter(240, 40)
        self.b3 = nn.Parameter(1, 40)
        self.m4 = nn.Parameter(40, 10)
        self.b4 = nn.Parameter(1, 10)
        self.alpha = 0.05

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.m1), self.b1))
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.m2), self.b2))
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.m3), self.b3))
        x = nn.AddBias(nn.Linear(x, self.m4), self.b4)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        allAcc = []
        while True:
            lossList = []
            for x, y in dataset.iterate_once(batch_size=100):
                loss = self.get_loss(x, y)
                lossList.append(nn.as_scalar(loss))
                grad_m1, grad_b1, grad_m2, grad_b2, grad_m3, grad_b3, grad_m4, grad_b4 = nn.gradients(
                    loss, [self.m1, self.b1, self.m2, self.b2, self.m3, self.b3, self.m4, self.b4]
                )
                self.m1.update(grad_m1, -self.alpha)
                self.b1.update(grad_b1, -self.alpha)
                self.m2.update(grad_m2, -self.alpha)
                self.b2.update(grad_b2, -self.alpha)
                self.m3.update(grad_m3, -self.alpha)
                self.b3.update(grad_b3, -self.alpha)
                self.m4.update(grad_m4, -self.alpha)
                self.b4.update(grad_b4, -self.alpha)
            validAcc = dataset.get_validation_accuracy()
            allAcc.append(validAcc)
            if len(allAcc) > 3 and allAcc[-1] < allAcc[-3]:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.alpha = 0.03
        self.w1 = nn.Parameter(self.num_chars, 512)
        self.b1 = nn.Parameter(1, 512)
        self.w2 = nn.Parameter(self.num_chars, 512)
        self.h = nn.Parameter(512, 512)
        self.b2 = nn.Parameter(1, 512)
        self.w3 = nn.Parameter(512, len(self.languages))
        self.b3 = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h_i = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w1), self.b1))
        for char in xs[1:]:
            h_i = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(char, self.w2), nn.Linear(h_i, self.h)), self.b2))
        output = nn.AddBias(nn.Linear(h_i, self.w3), self.b3)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        loss = float("inf")
        valid_acc = 0
        while valid_acc < 0.85:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grad_w1, grad_b1, grad_w2, grad_h, grad_b2, grad_w3, grad_b3 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.h, self.b2, self.w3, self.b3])
                self.w1.update(grad_w1, -self.alpha)
                self.b1.update(grad_b1, -self.alpha)
                self.w2.update(grad_w2, -self.alpha)
                self.h.update(grad_h, -self.alpha)
                self.b2.update(grad_b2, -self.alpha)
                self.w3.update(grad_w3, -self.alpha)
                self.b3.update(grad_b3, -self.alpha)
            valid_acc = dataset.get_validation_accuracy()


class Attention(object):
    def __init__(self, layer_size, block_size):
        """
        Initializes the Attention layer.

        Arguments:
            layer_size: The dimensionality of the input and output vectors.
            block_size: The size of the block for the causal mask (used to apply causal attention).

        We initialize the weight matrices (K, Q, and V) using random normal distributions.
        The causal mask is a lower triangular matrix (a matrix of zeros above the diagonal, ones on and below the diagonal).
        """

        self.k_weight = np.random.randn(layer_size, layer_size)
        self.q_weight = np.random.randn(layer_size, layer_size)
        self.v_weight = np.random.randn(layer_size, layer_size)

        # Create the causal mask using numpy
        self.mask = np.tril(np.ones((block_size, block_size)))

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to the input tensor. This includes computing the query, key, and value matrices,
        calculating the attention scores, applying the causal mask, and then generating the output.

        Arguments:
            input: The input tensor of shape (batch_size, block_size, layer_size).

        Returns:
            output: The output tensor after applying the attention mechanism to the input.

        Remark: remember to use the causal mask and nn.softmax (in nn.py) will be helpful.
        """

        B, T, C = input.shape

        """YOUR CODE HERE"""

