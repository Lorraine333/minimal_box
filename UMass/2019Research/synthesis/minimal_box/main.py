from __future__ import print_function
import utils
import numpy as np
import tensorflow as tf
import softbox as soft_model

flags = tf.app.flags
FLAGS = flags.FLAGS


def define_placeholder():
    placeholder = {}
    placeholder['t1_idx_placeholder'] = tf.placeholder(tf.int32, shape=[None]) # batch_size
    placeholder['t2_idx_placeholder'] = tf.placeholder(tf.int32, shape=[None]) # batch_size
    placeholder['label_placeholder'] = tf.placeholder(tf.float32, shape=[None]) # batch_size
    placeholder['marginal_label_placeholder'] = tf.placeholder(tf.float32, shape=[None]) # label_size
    return placeholder

def fill_feed_dict(data_set, marginal_prob, placeholder, batch_size):
    t1_idx, t2_idx, labels = data_set.next_batch(batch_size)
    feed_dict = {
        placeholder['t1_idx_placeholder']: t1_idx,
        placeholder['t2_idx_placeholder']: t2_idx,
        placeholder['label_placeholder']: labels,
        placeholder['marginal_label_placeholder']: marginal_prob
    }
    return feed_dict

def run_training():

    # read data set is a one time thing, so even it takes a little bit longer, it's fine.
    data_sets = utils.read_data_sets(FLAGS)
    train_data = data_sets.train

    with tf.Graph().as_default():
        print('Build Model...')
        placeholder = define_placeholder()
        model = soft_model.tf_model(placeholder, FLAGS)
        neg_log_prob = model.neg_log_prob
        train_op = model.training(model.loss, FLAGS.epsilon, FLAGS.learning_rate)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(FLAGS.max_steps):
            train_feed_dict = fill_feed_dict(train_data, data_sets.marginal_prob, placeholder, FLAGS.batch_size)
            _ , train_cond_loss, train_marg_loss, train_reg_loss, train_loss_value = sess.run([
                train_op, model.cond_loss, model.marg_loss, model.regularization, model.loss], feed_dict=train_feed_dict)
            if (step%(FLAGS.print_every) == 0) and step > 1:
                print('='*100)
                print('Epoch: %.1f Conditional loss: %.5f, Marginal loss: %.5f , Regularization loss: %.5f, Total loss: %.5f'
                      % (train_data._epochs_completed ,train_cond_loss, train_marg_loss, train_reg_loss, train_loss_value))
                neg_log = sess.run(neg_log_prob, feed_dict = train_feed_dict)
                true_label = train_feed_dict[placeholder['label_placeholder']]
                correlation = np.corrcoef(np.exp(-neg_log), true_label)[0, 1]
                print('Conditional Correlation: %.5f' % correlation)
                log_prob = sess.run(model.predicted_marginal_logits, feed_dict = train_feed_dict)
                marg_label = train_feed_dict[placeholder['marginal_label_placeholder']]
                marg_correlation = np.corrcoef(np.exp(log_prob), marg_label)[0, 1]
                print('Marginal Correlation: %.5f' % marg_correlation)


def main(argv):
    run_training()

if __name__ == '__main__':
    """dataset parameters"""
    flags.DEFINE_string('cond_file', '/Users/lorraine/UMass/2019Research/synthesis/script/cond_prob_no_zero.txt',
                        'conditional training file')
    flags.DEFINE_string('marg_file', '/Users/lorraine/UMass/2019Research/synthesis/script/marg_prob.txt',
                        'marginal training file')
    flags.DEFINE_integer('label_size', 80, 'number of labels')

    """optimization parameters"""
    flags.DEFINE_float('learning_rate', 1e-1, 'Initial learning rate.')
    flags.DEFINE_float('epsilon', 1e-8, 'Optimizer epsilon')

    """loss parameters"""
    flags.DEFINE_float('cond_weight', 0.9, 'weight on conditional prob loss')
    flags.DEFINE_float('marg_weight', 0.1, 'weight on marginal prob loss')
    flags.DEFINE_float('reg_weight', 0.0001, 'regularization parameter for universe')
    flags.DEFINE_string('regularization_method', 'delta', 'method to regularizing the embedding, either delta'
                                                                  'or universe_edge')

    """training parameters"""
    flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
    flags.DEFINE_integer('batch_size', 5418, 'Batch size. Must divide evenly into the dataset sizes.')
    flags.DEFINE_integer('print_every', 20, 'Every 20 step, print out the evaluation results')
    flags.DEFINE_integer('embed_dim', 10, 'word embedding dimension')

    tf.app.run()