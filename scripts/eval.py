#!/usr/bin/python
import os
import tensorflow as tf
from conv_model import ConvModel_270x480
import argparse
import scipy
import scipy.misc
import random
from data_loader import DataLoader
import math
import matplotlib.pyplot as plt
import time
import cv2
from cv_bridge import CvBridge
import shutil

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
    parser.add_argument('--dataset_index', type=str, default="",
                        help='The directory containing the training data.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Storing debug information for TensorBoard.')
    parser.add_argument('--logdir', type=str, default="logs",
                        help='Directory for log files.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--keep_prob', type=float, default=KEEP_PROB,
                        help='Dropout keep probability.')
    parser.add_argument('--dump_feature_map', type=bool, default=False,
                        help='dump feature map.')
    parser.add_argument('--dump_model_vis', type=bool, default=False,
                        help='dump model visualization image.')        
    parser.add_argument('--l2_reg', type=float,
                        default=L2_REG)
    parser.add_argument('--up_crop', type=int, default=UP_CROP)
    parser.add_argument('--alpha', type=int, default=0.9)
    parser.add_argument('--large_rmse_thresh', type=float, default=1.0)

    return parser.parse_args()


def render(figure_folder, model_path, curr_rolls, ys_vals, y_vals, smooth_rolls, rmse):
    xvals = [x for (x, y) in enumerate(ys_vals)]
    plt.clf()
    plt.plot(xvals, y_vals, label = 'pred_y', color='orange')    
    plt.plot(xvals, curr_rolls, label = 'curr_pose_roll', color='blue')
    plt.plot(xvals, ys_vals, label = "ground truth", color='green')
    plt.plot(xvals, smooth_rolls, label = 'smooth_pred_y', color='red')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.legend()
    fig = plt.gcf()
    fig.suptitle(model_path + ", rmse=" + str(rmse))
    
    ts = time.time()
    figure_path = figure_folder + '/eval_' + str(ts) + '.png'
    print 'save figure to', figure_path
    fig.set_size_inches(24, 16)
    plt.savefig(figure_path)

def dump_feature_map(data_path, orig_image, fmap_name, fmap):
    plt.clf()
    rows = (fmap.shape[-1] + 7) / 8
    figure_folder = data_path + '/feature_map/' + fmap_name
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig = plt.gcf()
    fig.suptitle(fmap_name + '/' + os.path.basename(orig_image))
    
    for i in range(fmap.shape[-1]):
        ax = plt.subplot(rows, 8, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(fmap[0, :, :, i], cmap='gray')
        
    output_path = figure_folder + '/' + os.path.basename(orig_image)
    fig.set_size_inches(24, 16)    
    plt.savefig(output_path)

def visualize_model_output(data_path, orig_image_path, curr_roll, target_roll, pred_roll, smooth_pred_roll, is_important):
    output_folder = data_path + '/model_vis'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = output_folder + '/' + os.path.basename(orig_image_path)
    raw_image = cv2.imread(orig_image_path)
    cv2.putText(raw_image, 'frame: %s' % os.path.basename(orig_image_path), (10, 26), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
    cv2.putText(raw_image, 'imu roll: %.2f' % float(curr_roll), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
    cv2.putText(raw_image, 'target pose roll: %.2f' % float(target_roll), (10, 74), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
    cv2.putText(raw_image, 'orig model roll: %.2f' % float(pred_roll), (10, 98), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
    cv2.putText(raw_image, 'smoothed model roll: %.2f' % float(smooth_pred_roll), (10, 122), cv2.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
    cv2.imwrite(output_path, raw_image)

    important_output_folder = data_path + '/model_important_vis'
    if not os.path.exists(important_output_folder):
        os.makedirs(important_output_folder)
    if is_important:
        important_output_path = important_output_folder + '/' + os.path.basename(orig_image_path)
        cv2.imwrite(important_output_path, raw_image)
    
def main():
    args = get_arguments()
    data_loader = DataLoader(args.dataset_index, args.data_file, args.up_crop)
    data_loader.load_for_eval()
    sess = tf.Session()
    # summary_writer = tf.summary.FileWriter(args.logdir, sess.graph_def)
    # summary_writer = tf.compat.v1.summary.FileWriter(args.logdir, sess.graph)
    
    model = ConvModel_270x480()
    train_vars = tf.trainable_variables()
    # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))) # + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * args.l2_reg
    loss = tf.losses.mean_squared_error(labels=model.y_, predictions=model.y)
    
    # train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss=loss)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    if args.model_path is not None:
        saver.restore(sess, args.model_path)

    min_loss = 1.0

    start_step = 0
    output_file = open(args.output_file, 'w')
    output_folder = os.path.dirname(args.output_file)
    rmse_error = 0.0
    ys_vals = []
    y_vals = []
    smooth_rolls = []
    curr_rolls = []
    large_rmse_count = 0
    total_sample_count = 0
    for train_bag_samples in data_loader.train_bags:
        data_path = train_bag_samples[0]
        bag_ys_vals = []
        bag_y_vals = []
        bag_rmse_error = 0.0
        bag_curr_rolls = []
        smooth_roll = None
        bag_smooth_rolls = []
        important_output_folder = data_path + '/model_important_vis'
        if os.path.exists(important_output_folder):
            shutil.rmtree(important_output_folder)
        
        for i in range(len(train_bag_samples[1])):
            xs, ys, x_name, curr_roll = data_loader.load_eval_batch(train_bag_samples, i)
            # train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob}, session=sess)
            bag_ys_vals.append(ys[0][0])
            ys_vals.append(ys[0][0])
            y, W_fc5, h_conv1, h_conv2, h_conv3, h_conv4, h_conv5, h_conv6 = sess.run([model.y, model.W_fc5, model.h_conv1, model.h_conv2, model.h_conv3, model.h_conv4, model.h_conv5, model.h_conv6], feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob})

            bag_y_vals.append(y[0][0])
            y_vals.append(y[0][0])
            # h_fc4 = sess.run(model.h_fc4, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob})
            pred_roll = y[0][0]
            target_roll = ys[0][0]
            bag_rmse_error += (pred_roll - target_roll) * (pred_roll - target_roll)
            rmse_error += (pred_roll - target_roll) * (pred_roll - target_roll)
            bag_curr_rolls.append(curr_roll[0])
            curr_rolls.append(curr_roll[0])
            
            # print len(y)
            if smooth_roll is None:
                smooth_roll = y[0][0]
            else:
                smooth_roll = args.alpha * smooth_roll + (1.0 - args.alpha) * y[0][0]
            bag_smooth_rolls.append(smooth_roll)
            smooth_rolls.append(smooth_roll)
            for k in range(len(y)):
                output_file.write("%s\t%f\t%f\t%f\t%f\n" % (x_name[k], curr_roll[0], target_roll, pred_roll, smooth_roll))
            # print W_fc5
            if abs(pred_roll - target_roll) > args.large_rmse_thresh:
                large_rmse_count += 1
                is_large_rmse_sample = True
            else:
                is_large_rmse_sample = False
            if args.dump_feature_map:
                dump_feature_map(data_path, x_name[0], 'h_conv1', h_conv1)
                dump_feature_map(data_path, x_name[0], 'h_conv2', h_conv2)
                dump_feature_map(data_path, x_name[0], 'h_conv3', h_conv3)
                dump_feature_map(data_path, x_name[0], 'h_conv4', h_conv4)
                dump_feature_map(data_path, x_name[0], 'h_conv5', h_conv5)
                # dump_feature_map(data_path, x_name[0], 'h_conv6', h_conv6)
                print 'finish dump:', x_name[0]
            if args.dump_model_vis:
                visualize_model_output(data_path, x_name[0], curr_roll[0], target_roll, pred_roll, smooth_roll, is_large_rmse_sample)

            total_sample_count += 1
            if i % 100 == 0:
                print "processed %d samples" % i                
        bag_rmse_error = math.sqrt(bag_rmse_error / len(bag_ys_vals))
        print "%s\t%f" % (data_path, bag_rmse_error)
        render(data_path, args.model_path, bag_curr_rolls, bag_ys_vals, bag_y_vals, bag_smooth_rolls, bag_rmse_error)

    rmse_error = math.sqrt(rmse_error / len(data_loader.train_xs))
    print "rmse: ", rmse_error
    print "large rmse count: ", large_rmse_count, ", ratio: ", float(large_rmse_count) / float(total_sample_count)
    output_file.close()
    render(output_folder, args.model_path, curr_rolls, ys_vals, y_vals, smooth_rolls, rmse_error)
    plt.show()

if __name__ == '__main__':
    main()

