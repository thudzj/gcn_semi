from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP
import scipy.sparse as sp

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'nell', 'Dataset string.')  # 'cora' 2.0, 0.7, 3, 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')#200
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.') #16
flags.DEFINE_integer('seed', 123, 'Random seed.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).') #0.5
flags.DEFINE_float('p1', 2.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('p2', 0.7, 'Dropout rate (1 - keep probability).')
#flags.DEFINE_float('p3', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight for L2 loss on embedding matrix.') #5e-4
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('epsilon', 0.03, "Norm length for (virtual) adversarial training ")
flags.DEFINE_integer('num_power_iterations', 3, "The number of power iterations")
flags.DEFINE_float('xi', 1e-6, "Small constant for finite difference")

# Set random seed
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
# Load data
is_sparse = True
nbrs, support, support_test, features, labels, train_mask, val_mask, test_mask = load_data(FLAGS.dataset, is_sparse)

if is_sparse:
    feature_size = features[2][1]
    N = features[2][0]
else:
    feature_size = features.shape[1]
    N = features.shape[0]

# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32) if is_sparse else tf.placeholder(tf.float32, shape=(N, feature_size)),
    'norm_mtx': tf.placeholder(tf.float32, shape=(N, N)),
    'labels': tf.placeholder(tf.float32, shape=(N, labels.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
    'adv_mask1': tf.placeholder(tf.int32),
    'adv_mask2': tf.placeholder(tf.int32),
}

# Create model
model = GCN(placeholders, is_sparse, input_dim=feature_size, multitask=(FLAGS.dataset == 'ppi'), logging=True)

# Initialize session
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders, nbrs)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    micro, macro = calc_f1(outs_val[2][mask], labels[mask], FLAGS.dataset == 'ppi')
    return outs_val[0], outs_val[1], micro, macro, (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, labels, train_mask, placeholders, nbrs)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # tmp = np.zeros(features.shape[0])
    # tmp[0] = 1
    # feed_dict.update({placeholders['adv_mask1']: tmp})
    # rrr = sess.run(model.ppp, feed_dict=feed_dict)
    # print(rrr)
    # ggg = sp.coo_matrix(rrr[0], (rrr[1][:,0], rrr[1][:,1]))
    # print(ggg)
    # exit(1)

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # Validation
    cost, acc, micro, macro, duration = evaluate(features, support_test, labels, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))


    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, micro, macro, test_duration = evaluate(features, support_test, labels, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
