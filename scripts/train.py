#!/usr/bin/python
import os
import tensorflow as tf
from conv_model import ConvModel_270x480
import argparse
import scipy
import scipy.misc
import random
from data_loader import DataLoader

BATCH_SIZE = 100
CHECKPOINT_EVERY = 100
NUM_STEPS = int(1e5)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 1e-3
KEEP_PROB = 0.8
L2_REG = 0
EPSILON = 0.001
MOMENTUM = 0.9
UP_CROP = 60

def get_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch.')
    parser.add_argument('--data_file', type=str, default="",
                        help='The data file training data.')        
    parser.add_argument('--dataset_index', '--data', type=str, default="",
                        help='The directory containing the training data.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Storing debug information for TensorBoard.')
    parser.add_argument('--logdir', type=str, default="logs",
                        help='Directory for log files.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--keep_prob', type=float, default=KEEP_PROB,
                        help='Dropout keep probability.')
    parser.add_argument('--l2_reg', type=float,
                        default=L2_REG)
    parser.add_argument('--skip_frames', type=float,
                        default=0)
    parser.add_argument('--up_crop', type=int, default=UP_CROP)
    
    return parser.parse_args()


def main():
    args = get_arguments()
    data_loader = DataLoader(args.dataset_index, args.data_file, args.up_crop)
    data_loader.load_for_train(args.skip_frames)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    
    sess = tf.Session()
    # summary_writer = tf.compat.v1.summary.FileWriter(args.logdir, sess.graph)
    # summary_writer = tf.summary.create_file_writer(args.logdir)
    
    model = ConvModel_270x480()
    train_vars = tf.trainable_variables()
    # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))) # + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * args.l2_reg
    loss = tf.losses.mean_squared_error(labels=model.y_, predictions=model.y)
    tf.summary.scalar('loss', loss)
    
    # tf.scalar_summary("loss", loss)
    # merged_summary_op = tf.merge_all_summaries()
    # summary_writer = tf.train.SummaryWriter(args.logdir, graph=tf.get_default_graph())
    
    # train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss=loss)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    min_loss = 1.0

    start_step = 0
    summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)
    merged = tf.summary.merge_all()
    
    for i in range(start_step, start_step + args.num_steps):
        xs, ys, x_name = data_loader.load_train_batch(args.batch_size)
        train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob}, session=sess)
        train_error = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0}, session=sess)
        print("Step %d, train loss %g" % (i, train_error))

        summary_str, y, W_fc5, b_fc5 = sess.run([merged, model.y, model.W_fc5, model.b_fc5], feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob})
        # h_fc4 = sess.run(model.h_fc4, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob})
        # print summary_str
        # print len(y)
        for k in range(len(y)):
            print x_name[k], ys[k], y[k]


        if i % 10 == 0:
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

            # xs, ys = data_reader.load_val_batch(args.batch_size)
            # val_error = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
            # print("Step %d, val loss %g" % (i, val_error))
            if i > 0 and i % args.checkpoint_every == 0:
                checkpoint_path = os.path.join(args.logdir, "model-step-%d.ckpt" % (i))
                filename = saver.save(sess, checkpoint_path)
                print("Model saved in file: %s" % filename)
                print(W_fc5, b_fc5)
                
                # summary = sess.run(tf.summary.scalar('loss', loss))
                # summary = tf.compat.v1.summary.scalar('loss', train_error) 
                # summary_writer.add_summary(summary, i)
                
if __name__ == '__main__':
    main()

