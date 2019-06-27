from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import utils
from box import MyBox
import tensorflow as tf

my_seed = 20180112
tf.set_random_seed(my_seed)

class tf_model(object):
    def __init__(self, placeholder, FLAGS):
        self.label_size = FLAGS.label_size
        self.embed_dim = FLAGS.embed_dim
        self.cond_weight = FLAGS.cond_weight
        self.marg_weight = FLAGS.marg_weight
        self.reg_weight = FLAGS.reg_weight
        self.regularization_method = FLAGS.regularization_method
        self.temperature = 1.0

        self.t1x = placeholder['t1_idx_placeholder'] #[batch_size]
        self.t2x = placeholder['t2_idx_placeholder'] #[batch_size]
        self.label = placeholder['label_placeholder'] #[batch_size]
        self.marginal_label = placeholder['marginal_label_placeholder'] # [label_size]

        # self.t1x = tf.Print(self.t1x, [tf.shape(self.t1x), tf.shape(self.t2x), tf.shape(self.label), tf.shape(self.marginal_label)], 'shape')

        """Initiate box embeddings"""
        self.min_embed, self.delta_embed = self.init_word_embedding()
        """For training"""
        self.t1_box = self.get_word_embedding(self.t1x)
        self.t2_box = self.get_word_embedding(self.t2x)
        conditional_logits, self.meet_box, self.disjoint,\
        self.nested, self.overlap_volume, self.rhs_volume = self.get_conditional_probability(self.t1_box, self.t2_box)
        evaluation_logits, _, _, _, _, _ = self.get_conditional_probability(self.t1_box, self.t2_box)
        self.neg_log_prob = -evaluation_logits

        """get conditional probability loss"""
        cond_pos_loss = tf.multiply(conditional_logits, self.label)
        cond_neg_loss = tf.multiply(tf.log(1-tf.exp(conditional_logits)+1e-10), 1-self.label)
        cond_loss = -tf.reduce_mean(cond_pos_loss+ cond_neg_loss)
        self.cond_loss = self.cond_weight * cond_loss

        """model marg prob loss"""
        if self.marg_weight > 0.0:
            # prediction
            self.max_embed = self.min_embed + tf.exp(self.delta_embed)
            self.universe_min = tf.reduce_min(self.min_embed, axis=0, keepdims=True)
            self.universe_max = tf.reduce_max(self.max_embed, axis=0, keepdims=True)
            self.universe_volume = self.volume_calculation(MyBox(self.universe_min, self.universe_max))
            self.box_volume = self.volume_calculation(MyBox(self.min_embed, self.max_embed))
            self.predicted_marginal_logits = tf.log(self.box_volume) - tf.log(self.universe_volume)
            # marginal loss
            marg_pos_loss = tf.multiply(self.predicted_marginal_logits, self.marginal_label)
            marg_neg_loss = tf.multiply(tf.log(1-tf.exp(self.predicted_marginal_logits)+1e-10), 1-self.marginal_label)
            self.marg_loss = -tf.reduce_mean(marg_pos_loss+marg_neg_loss)
            self.marg_loss *= self.marg_weight
        else:
            self.marg_loss = tf.constant(0.0)

        """model regurlization"""
        if self.regularization_method == 'universe_edge' and self.reg_weight>0.0:
            self.regularization = self.reg_weight * tf.reduce_mean(
                tf.nn.softplus(self.universe_max - self.universe_min))
        elif self.regularization_method == 'delta' and self.reg_weight>0.0:
            self.regularization = self.reg_weight * tf.reduce_mean(
                tf.square(tf.exp(self.delta_embed)))
        else:
            self.regularization = tf.constant(0.0)


        """model final loss"""
        self.loss = self.cond_loss + self.marg_loss + self.regularization


    def volume_calculation(self, mybox):
        return tf.reduce_prod(tf.nn.softplus((mybox.max_embed - mybox.min_embed)/
                                             self.temperature)*self.temperature, axis=-1)

    def init_embedding_scale(self):
        # softbox delta log init
        # min_lower_scale, min_higher_scale = 1e-4, 0.9
        # delta_lower_scale, delta_higher_scale = -1.0, -0.1
        min_lower_scale, min_higher_scale = 1e-4, 0.9
        delta_lower_scale, delta_higher_scale = -0.1, 0
        return min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale

    def init_word_embedding(self):
        min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale = self.init_embedding_scale()
        min_embed = tf.Variable(
            tf.random_uniform([self.label_size, self.embed_dim],
                              min_lower_scale, min_higher_scale, seed=my_seed), trainable=True, name='word_embed')
        delta_embed = tf.Variable(
            tf.random_uniform([self.label_size, self.embed_dim],
                              delta_lower_scale, delta_higher_scale, seed=my_seed), trainable=True, name='delta_embed')
        return min_embed, delta_embed

    def get_word_embedding(self, idx):
        """read word embedding from embedding table, get unit cube embeddings"""
        min_embed = tf.nn.embedding_lookup(self.min_embed, idx)
        delta_embed = tf.nn.embedding_lookup(self.delta_embed, idx) # [batch_size, embed_size]
        max_embed = min_embed + tf.exp(delta_embed)
        t1_box = MyBox(min_embed, max_embed)
        return t1_box

    def get_conditional_probability(self, t1_box, t2_box):
        _, meet_box, disjoint = utils.calc_join_and_meet(t1_box, t2_box)
        nested = utils.calc_nested(t1_box, t2_box, self.embed_dim)
        """get conditional probabilities"""
        overlap_volume = self.volume_calculation(meet_box)
        rhs_volume = self.volume_calculation(t1_box)
        conditional_logits = tf.log(overlap_volume+1e-10) - tf.log(rhs_volume+1e-10)
        return conditional_logits, meet_box, disjoint, nested, overlap_volume, rhs_volume


    def training(self, loss, epsilon, learning_rate):
        tf.summary.scalar(loss.op.name, loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon = epsilon, use_locking=True)
        train_op = optimizer.minimize(loss)
        return train_op

