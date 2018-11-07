import numpy as np
import tensorflow as tf
from collections import Counter

train_x = np.load('data/processed/np_train_x.npy')
train_y = np.load('data/processed/np_train_y.npy')
# train_x_one_hot = np.load("data/processed/np_train_y_one_hot.npy")
test_x = np.load('data/processed/np_test_x.npy')
test_y = np.load('data/processed/np_test_y.npy')
# test_x_one_hot = np.load("data/processed/np_test_x_one_hot.npy")
print("### train_x shape:", train_x.shape, "###")
print("### test_x shape:", test_x.shape, "###")
print("### train_y shape:", train_y.shape, Counter(train_y), "###")
print("### test_y shape:", test_y.shape, Counter(test_y), "###")
print("### Process --- data load --- finished ###")

train_data_count = train_x.shape[0]
test_data_count = test_x.shape[0]
n_steps = train_x.shape[1]
n_input = train_x.shape[2]

n_hidden = 32  # Hidden layer num of features
n_classes = 6  # Total classes (should go up, or should go down)

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_steps = 1000  # Loop 1000 times
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training


def LSTM_RNN(_X, _weights, _biases):
    _X = tf.transpose(_X, [1, 0, 2])  # (128, ?, 9)
    print("# transpose shape: ", _X.shape)    # transpose shape:  (128, ?, 9)
    _X = tf.reshape(_X, [-1, n_input])  # (n_step*batch_size, n_input)
    print("# reshape shape: ", _X.shape)    # reshape shape:  (?, 9)
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    print("# matmul shape: ", _X.shape)     # matmul shape:  (?, 32)
    _X = tf.split(_X, n_steps, 0)  # n_steps * (batch_size, n_hidden)
    print("# split shape: ", np.array(_X).shape)    # split shape:  (128,)

    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    print("# cell_1 shape: ", lstm_cell_1.state_size)   # cell_1 shape:  LSTMStateTuple(c=32, h=32)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    print("# cell_2 shape: ", lstm_cell_2.state_size)   # cell_2 shape:  LSTMStateTuple(c=32, h=32)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    print("# multi cells shape: ", lstm_cells.state_size)
    # multi cells shape:  (LSTMStateTuple(c=32, h=32), LSTMStateTuple(c=32, h=32))

    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    print("# outputs & states shape: ", np.array(outputs).shape, np.array(states).shape)
    # outputs & states shape:  (128,) (2, 2)

    lstm_last_output = outputs[-1]  # N to 1
    print("# last output shape: ", lstm_last_output.shape)  # last output shape:  (?, 32)
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
y = tf.placeholder(tf.float32, [None, n_classes], name='y')

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # step = 1
    for step in range(training_steps):
        batch_xs = extract_batch_size(train_x, step, batch_size)
        batch_ys = one_hot(extract_batch_size(train_y, step, batch_size))

        # Fit training using batch data
        _, loss, acc = sess.run([optimizer, cost, accuracy],
                                feed_dict={x: batch_xs, y: batch_ys})
        if (step+1) % 10 == 0:
            print("Step", (step+1), "| Train loss:", loss,
                  "| Train accuracy:", acc)
        if (step+1) % 50 == 0:
            loss, acc = sess.run([cost, accuracy],
                                 feed_dict={x: test_x, y: one_hot(test_y)})
            print("Step", (step+1), "| Test loss:", loss,
                  "| Test accuracy:", acc)

# 2018/11/7
# Step 1000 | Train loss: 0.29677442 | Train accuracy: 0.9773333
# Step 1000 | Test loss: 0.6675542 | Test accuracy: 0.88089585
