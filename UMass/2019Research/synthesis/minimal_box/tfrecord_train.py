import argparse
import tensorflow as tf
from softbox_model import model_fn

def cond_input_fn(filename, batch_size, shuffle, epoch):

    def parse_cond_feature(raw_record):
        """"
        return a tuple of feature and label
        feature is a dictionay of three tensors, first two tensors with shape [example_size]
        third one with shape [label_size]
        label is a tensor with shape [example_size]
        """""
        # Define features
        feature_map={
            'u': tf.FixedLenFeature([5418], dtype=tf.int64),
            'v': tf.FixedLenFeature([5418], dtype=tf.int64),
            'prob': tf.FixedLenFeature([5418], dtype=tf.float32)}

        features = tf.parse_single_example(raw_record, feature_map)
        return_features = {
            'idx': tf.zeros_like(features['u']),
            'term1': features['u'],
            'term2': features['v']}
        labels = {'prob': features['prob']}
        return return_features, labels

    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse_cond_feature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(batch_size * 10)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def marg_input_fn(filename, batch_size, shuffle, epoch):
    def parse_marg_feature(raw_record):
        """"
        return a tuple of feature and label
        feature is a dictionay of three tensors, first two tensors with shape [example_size]
        third one with shape [label_size]
        label is a tensor with shape [example_size]
        """""
        # Define features

        feature_map={
            'marg_idx' : tf.FixedLenFeature([80], dtype=tf.int64),
            'marg_prob': tf.FixedLenFeature([80], dtype=tf.float32)}
        features = tf.parse_single_example(raw_record, feature_map)

        return_features = {'idx': features['marg_idx'],
                           'term1': tf.zeros_like(features['marg_idx']),
                           'term2': tf.zeros_like(features['marg_idx'])}
        labels = {'prob': features['marg_prob']}

        return return_features, labels

    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse_marg_feature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(batch_size * 10)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument('--cond_file', type=str, default='/Users/lorraine/UMass/2019Research/synthesis/script/cond.tfrecord')
    parser.add_argument('--marg_file', type=str, default='/Users/lorraine/UMass/2019Research/synthesis/script/marg.tfrecord')
    parser.add_argument('--model_dir', type=str, default='./tmp')
    parser.add_argument('--label_size', type=int, default=80)
    parser.add_argument('--learning_rate', type=float, default=1e-2)

    parser.add_argument('--cond_weight', type=float, default=0.9)
    parser.add_argument('--marg_weight', type=float, default=0.1)
    parser.add_argument('--reg_weight', type=float, default=0.001)
    parser.add_argument('--regularization_method', type=str, default='delta')

    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--embed_dim', type=int, default=10)

    args = parser.parse_args()

    config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                    save_checkpoints_steps=20,
                                    )

    myestimator = tf.estimator.Estimator(
        model_fn=model_fn, config=config, params={
            'learning_rate': args.learning_rate,
            'label_size': args.label_size,
            'cond_weight': args.cond_weight,
            'marg_weight': args.marg_weight,
            'reg_weight': args.reg_weight,
            'regularization_method': args.regularization_method,
            'embed_dim': args.embed_dim
        })

    for num_train_epochs in range(1000):
        myestimator.train(
            input_fn=lambda: cond_input_fn(args.cond_file, 5418, False, 1),
            max_steps=args.max_steps
        )
        myestimator.train(
            input_fn=lambda: marg_input_fn(args.marg_file, 80, True, 1),
            max_steps=args.max_steps
        )
        myestimator.evaluate(
            input_fn=lambda: cond_input_fn(args.cond_file, 5418, False, 1),
            steps=args.max_steps)

        myestimator.evaluate(
            input_fn=lambda: marg_input_fn(args.marg_file, 80, True, 1),
            steps=args.max_steps)
