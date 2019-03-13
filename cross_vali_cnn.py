# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.model_selection import KFold, cross_val_score
import csv
from sklearn.utils import shuffle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

window_size = 500
threshold = 60

# Parameters
learning_rate = 0.0001
training_iters = 2000
batch_size = 64

# Import WiFi Activity data
# csv_convert(window_size,threshold)
from cross_vali_input_data import csv_import, DataSet

# Network Parameters
n_input = 90  # WiFi activity data input (img shape: 90*window_size)
n_steps = window_size  # timesteps
#n_hidden = 200 # hidden layer num of features original 200
n_classes = 7  # WiFi activity total classes
display_step = 100

# Output folder
OUTPUT_FOLDER_PATTERN = "CNN_LR{0}_BATCHSIZE{1}/"
output_folder = OUTPUT_FOLDER_PATTERN.format(learning_rate, batch_size)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# tf Graph input
x = tf.placeholder("float", [None, n_steps,  n_input])
y = tf.placeholder("float", [None, n_classes])
x_image = tf.reshape(x, [-1, 500,90,1])

w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

w_fc1 = weight_variable([125 * 23 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 125 * 23 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, n_classes])
b_fc2 = bias_variable([n_classes])

pred = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
cvscores = []
confusion_sum = [[0 for i in range(7)] for j in range(7)]

# data import
x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk, \
y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk = csv_import()


print(" bed =", len(x_bed), " fall=", len(x_fall), " pickup =", len(x_pickup), " run=", len(x_run), " sitdown=",
      len(x_sitdown), " standup=", len(x_standup), " walk=", len(x_walk))

# data shuffle
x_bed, y_bed = shuffle(x_bed, y_bed, random_state=0)
x_fall, y_fall = shuffle(x_fall, y_fall, random_state=0)
x_pickup, y_pickup = shuffle(x_pickup, y_pickup, random_state=0)
x_run, y_run = shuffle(x_run, y_run, random_state=0)
x_sitdown, y_sitdown = shuffle(x_sitdown, y_sitdown, random_state=0)
x_standup, y_standup = shuffle(x_standup, y_standup, random_state=0)
x_walk, y_walk = shuffle(x_walk, y_walk, random_state=0)

# k_fold
kk = 10

# Launch the graph
with tf.Session() as sess:
    for i in range(kk):

        # Initialization
        train_loss = []
        train_acc = []
        validation_loss = []
        validation_acc = []

        # Roll the data
        x_bed = np.roll(x_bed, int(len(x_bed) / kk), axis=0)
        y_bed = np.roll(y_bed, int(len(y_bed) / kk), axis=0)
        x_fall = np.roll(x_fall, int(len(x_fall) / kk), axis=0)
        y_fall = np.roll(y_fall, int(len(y_fall) / kk), axis=0)
        x_pickup = np.roll(x_pickup, int(len(x_pickup) / kk), axis=0)
        y_pickup = np.roll(y_pickup, int(len(y_pickup) / kk), axis=0)
        x_run = np.roll(x_run, int(len(x_run) / kk), axis=0)
        y_run = np.roll(y_run, int(len(y_run) / kk), axis=0)
        x_sitdown = np.roll(x_sitdown, int(len(x_sitdown) / kk), axis=0)
        y_sitdown = np.roll(y_sitdown, int(len(y_sitdown) / kk), axis=0)
        x_standup = np.roll(x_standup, int(len(x_standup) / kk), axis=0)
        y_standup = np.roll(y_standup, int(len(y_standup) / kk), axis=0)
        x_walk = np.roll(x_walk, int(len(x_walk) / kk), axis=0)
        y_walk = np.roll(y_walk, int(len(y_walk) / kk), axis=0)

        # data separation
        wifi_x_train = np.r_[
            x_bed[int(len(x_bed) / kk):], x_fall[int(len(x_fall) / kk):], x_pickup[int(len(x_pickup) / kk):], \
            x_run[int(len(x_run) / kk):], x_sitdown[int(len(x_sitdown)/kk):], x_standup[
                                                                                int(len(x_standup) / kk):], x_walk[int(
                len(x_walk) / kk):]]

        wifi_y_train = np.r_[
            y_bed[int(len(y_bed) / kk):], y_fall[int(len(y_fall) / kk):], y_pickup[int(len(y_pickup) / kk):], \
            y_run[int(len(y_run) / kk):], y_sitdown[int(len(y_sitdown)/kk):], y_standup[
                                                                                int(len(y_standup) / kk):], y_walk[int(
                len(y_walk) / kk):]]

        wifi_y_train = wifi_y_train[:, 1:]

        wifi_x_validation = np.r_[
            x_bed[:int(len(x_bed) / kk)], x_fall[:int(len(x_fall) / kk)], x_pickup[:int(len(x_pickup) / kk)], \
            x_run[:int(len(x_run) / kk)], x_sitdown[:int(len(x_sitdown)/kk)], x_standup[
                                                                                :int(len(x_standup) / kk)], x_walk[:int(
                len(x_walk) / kk)]]

        wifi_y_validation = np.r_[
            y_bed[:int(len(y_bed) / kk)], y_fall[:int(len(y_fall) / kk)], y_pickup[:int(len(y_pickup) / kk)], \
            y_run[:int(len(y_run) / kk)], y_sitdown[:int(len(y_sitdown)/kk)], y_standup[
                                                                                :int(len(y_standup) / kk)], y_walk[:int(
                len(y_walk) / kk)]]

        wifi_y_validation = wifi_y_validation[:, 1:]

        # data set
        wifi_train = DataSet(wifi_x_train, wifi_y_train)
        wifi_validation = DataSet(wifi_x_validation, wifi_y_validation)
        print(wifi_x_train.shape, wifi_y_train.shape, wifi_x_validation.shape, wifi_y_validation.shape)
        saver = tf.train.Saver()
        sess.run(init)
        step = 1

        # Keep training until reach max iterations
        while step < training_iters:
            batch_x, batch_y = wifi_train.next_batch(batch_size)
            x_vali = wifi_validation.images[:]
            y_vali = wifi_validation.labels[:]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            x_vali = x_vali.reshape((-1, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
            acc_vali = sess.run(accuracy, feed_dict={x: x_vali, y: y_vali, keep_prob : 1.0})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
            loss_vali = sess.run(cost, feed_dict={x: x_vali, y: y_vali, keep_prob:1.0})

            # Store the accuracy and loss
            train_acc.append(acc)
            train_loss.append(loss)
            validation_acc.append(acc_vali)
            validation_loss.append(loss_vali)

            if step % display_step == 0:
                print("Iter " + str(step) + ", Minibatch Training  Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + ", Minibatch Validation  Loss= " + \
                      "{:.6f}".format(loss_vali) + ", Validation Accuracy= " + \
                      "{:.5f}".format(acc_vali))
            step += 1

        # Calculate the confusion_matrix
        cvscores.append(acc_vali * 100)
        y_p = tf.argmax(pred, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_vali, y: y_vali, keep_prob:1.0})
        y_true = np.argmax(y_vali, 1)
        print(sk.metrics.confusion_matrix(y_true, y_pred))
        confusion = sk.metrics.confusion_matrix(y_true, y_pred)
        confusion_sum = confusion_sum + confusion

        # Save the Accuracy curve
        fig = plt.figure(2 * i - 1)
        plt.plot(train_acc)
        plt.plot(validation_acc)
        plt.xlabel("n_epoch")
        plt.ylabel("Accuracy")
        plt.legend(["train_acc", "validation_acc"], loc=4)
        plt.ylim([0, 1])
        plt.savefig((output_folder + "Accuracy_" + str(i) + ".png"), dpi=150)

        # Save the Loss curve
        fig = plt.figure(2 * i)
        plt.plot(train_loss)
        plt.plot(validation_loss)
        plt.xlabel("n_epoch")
        plt.ylabel("Loss")
        plt.legend(["train_loss", "validation_loss"], loc=1)
        plt.ylim([0, 2])
        plt.savefig((output_folder + "Loss_" + str(i) + ".png"), dpi=150)

        np.save(output_folder + "cnn_trainloss_"+ str(i) +".npy",np.array(train_loss))
        np.save(output_folder + "cnn_valloss_" + str(i) + ".npy",np.array(validation_loss))
        np.save(output_folder + "cnn_trainacc_"+ str(i) + ".npy", np.array(train_acc))
        np.save(output_folder + "cnn_valacc_" + str(i) + ".npy", np.array(validation_acc))
    print("Optimization Finished!")
    print("%.1f%% (+/- %.1f%%)" % (np.mean(cvscores), np.std(cvscores)))
    saver.save(sess, output_folder + "model.ckpt")

    # Save the confusion_matrix
    np.savetxt(output_folder + "confusion_matrix.txt", confusion_sum, delimiter=",", fmt='%d')
    np.savetxt(output_folder + "accuracy.txt", (np.mean(cvscores), np.std(cvscores)), delimiter=".", fmt='%.1f')
