import pickle as p
import numpy as np
import random
import tensorflow as tf
import numpy as np

# Intro to Computation Graph

@tf.function
def calc(a, b):
    node1 = tf.constant(a)
    node2 = tf.constant(b)
    node3 = tf.add(node1, node2)
    node4 = tf.subtract(node2, 1)
    node5 = tf.multiply(node3, node4)
    return node5

output = calc(3.0, 6.0)
print(f"Computation Graph\nOutput: {output}")


# Neural Networks in TF
test_input = [[0.5, 0.5]]

@tf.function
def sigmoid():
    return 0

n_input_nodes = 2
n_output_nodes = 1
# => many to one 

x = tf.constant(tf.float32)
w = tf.Variable(tf.ones((n_input_nodes, n_output_nodes)), dtype=tf.float32)
b = tf.Variable(tf.zeros(n_output_nodes), dtype=tf.float32)

# TODO: define 'z' and 'output'
z = tf.add(tf.multiply(x, w), b)
output = tf.sigmoid(z)
print(f"Neural Networks\nOutput: {output}")
