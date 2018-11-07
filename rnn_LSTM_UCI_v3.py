import numpy as np
import tensorflow as tf
# from tensorflow.contrib import rnn

train_x = np.load('data/processed/np_train_x.npy')
train_y = np.load("data/processed/np_train_y_one_hot.npy")
test_x = np.load('data/processed/np_test_x.npy')
test_y = np.load("data/processed/np_test_y_one_hot.npy")
print("# train_x shape:", train_x.shape, "###")
print("# test_x shape:", test_x.shape, "###")
print("# train_y shape:", train_y.shape, "###")
print("# test_y shape:", test_y.shape, "###")
print("# Process --- data load --- finished ###")

train_data_count = train_x.shape[0]
test_data_count = test_x.shape[0]
n_steps = train_x.shape[1]
n_input = train_x.shape[2]

n_hidden = 32
n_classes = 6

learning_rate = 0.0025
training_steps = 10000  # Loop 10000 times
batch_size = 1000
train_shuffle_index = np.arange(train_data_count)
np.random.shuffle(train_shuffle_index)


def lstm_rnn(_x, batch_size=batch_size, n_hidden=n_hidden):
    _x = tf.transpose(_x, [1, 0, 2])  # (128, ?, 9)
    print("# transpose shape: ", _x.shape)    # transpose shape:  (128, ?, 9)
    _x = tf.layers.dense(
        inputs=_x,
        units=n_hidden,
        activation=tf.nn.relu,
    )
    print("# relu shape: ", _x.shape)     # relu shape:  (128, ?, 32)

    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_1_drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=0.5)
    print("# cell_1 shape: ", lstm_cell_1.state_size)   # cell_1 shape:  LSTMStateTuple(c=32, h=32)

    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2_drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=0.5)
    print("# cell_2 shape: ", lstm_cell_2.state_size)   # cell_2 shape:  LSTMStateTuple(c=32, h=32)

    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1_drop, lstm_cell_2_drop], state_is_tuple=True)
    print("# multi cells shape: ", lstm_cells.state_size)
    # multi cells shape:  (LSTMStateTuple(c=32, h=32), LSTMStateTuple(c=32, h=32))

    initial_state = lstm_cells.zero_state(batch_size=batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=_x, initial_state=initial_state,
                                        dtype=tf.float32, time_major=True)
    print("# outputs & states shape: ", outputs.shape, np.array(states).shape)
    # outputs & states shape:  (128, batch_size=1000, 32) (2, 2)
    # tf.nn.dynamic_rnn
    lstm_last_output = outputs[-1]  # N to 1
    print("# last output shape: ", lstm_last_output.shape)  # last output shape:  (?, 32)

    lstm_last_output = tf.layers.dense(
        inputs=lstm_last_output,
        units=n_hidden,
        activation=tf.nn.relu
    )
    print("# fully connected shape: ", lstm_last_output.shape)  # fully connected shape:  (?, 32)

    prediction = tf.layers.dense(
        inputs=lstm_last_output,
        units=n_classes,
        activation=tf.nn.softmax
    )
    print("# prediction shape: ", prediction.shape)  # prediction shape:  (?, 6)
    return prediction


def extract_batch_size(_train, step, batch_size=batch_size, train_shuffle_index=train_shuffle_index):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = train_shuffle_index[((step - 1) * batch_size + i) % len(_train)]
        batch_s[i] = _train[index]

    return batch_s


x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
y = tf.placeholder(tf.float32, [None, n_classes], name='y')
batch_size_n = tf.placeholder(tf.int32, [])
pred = lstm_rnn(x, batch_size_n)

loss = - tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(training_steps):
        batch_xs = extract_batch_size(train_x, step)
        batch_ys = extract_batch_size(train_y, step)
        _, loss_train, acc_train = sess.run([optimizer, loss, accuracy],
                                            feed_dict={x: batch_xs, y: batch_ys, batch_size_n: batch_size})
        if (step+1) % 50 == 0:
            print("# Step", (step+1), "| Train loss:", loss_train,
                  "| Train accuracy:", acc_train)
        if (step+1) % 200 == 0:
            loss_test, acc_test = sess.run([loss, accuracy],
                                           feed_dict={x: test_x, y: test_y, batch_size_n: test_data_count})
            print("# Step", (step+1), "| Test loss:", loss_test,
                  "| Test accuracy:", acc_test)


# shuffle后train acc上升过程更稳定，不易出现掉幅很大的现象， test acc稳步上升
# Step 4000 | Train loss: 0.0138888545 | Train accuracy: 0.967
# Step 4000 | Test loss: 0.1128672 | Test accuracy: 0.8975229
