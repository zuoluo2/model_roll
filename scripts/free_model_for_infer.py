#!/usr/bin/python
import argparse
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util
from conv_model import ConvModel_270x480
import uff
import tensorrt as trt
import os
# from int8_calibrator import Int8Calibrator
# import faulthandler
# faulthandler.enable()

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 270

def getArgs():
    # input is the .tsv file generated from lane simulator when using --save_lane_result true
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--output_graph', type=str, default='', help='output absolute path for pb model file')
    parser.add_argument('--output_uff', type=str, default='', help='output absolute path for uff model file')
    parser.add_argument('--output_engine', type=str, default='', help='output absolute path for engine model file')
    parser.add_argument('--bit16', default=False, action='store_true')
    parser.add_argument('--bit8', default=False, action='store_true')

    args = parser.parse_args()

    return args

def freeze_engine_from_pb(args):
    max_batch_size = 1
    TRT_LOGGER = trt.Logger(trt.Logger.Severity.INFO)
    builder = trt.Builder(TRT_LOGGER)

    if args.bit16 and not args.bit8:
        print("Freezing 16 bit engine!")
        builder.fp16_mode = True
        builder.strict_type_constraints = True
    elif args.bit8:
        print("Freezing 8 bit engine!")
        builder.int8_mode = True
        calibrator = Int8Calibrator(
            input_size=[64, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
            calibration_data=args.calibration_data,
            # crop_area=(0, 150, 1072, 772),
            cache_file=args.calibration_cache,
            calibration_data_size=args.calibration_data_size)
        builder.int8_calibrator = calibrator
        builder.strict_type_constraints = True
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 31
    network = builder.create_network()
    pb_filename = args.output_graph
    uff_filename = args.output_uff
    outputs = ['output/roll']
    print("Loading pb from ", pb_filename)
    print("Save uff to ", uff_filename)
    uff_model = uff.from_tensorflow_frozen_model(pb_filename,
                                                 outputs,
                                                 preprocessor=None,
                                                 output_filename=uff_filename,
                                                 return_graph_info=True,
                                                 debug_mode=True,
                                                 quiet=False,
                                                 list_nodes=False)

    uff_parser = trt.UffParser()
    succ1 = uff_parser.register_input(
        name="input/image",
        shape=[3, IMAGE_HEIGHT, IMAGE_WIDTH])
    print("register input: ", succ1)
    for output in outputs:
        uff_parser.register_output(output)
    print("parsing uff model into engine model")
    uff_parser.parse(uff_filename, network)
    with builder.build_cuda_engine(network) as engine:
        with open(args.output_engine, "wb") as f:
            print("Writing engine model to %s" % args.output_engine)
            f.write(engine.serialize())


def main():
    args = getArgs()
    sess = tf.Session()
    # summary_writer = tf.summary.FileWriter(args.logdir, sess.graph_def)
    # summary_writer = tf.compat.v1.summary.FileWriter(args.logdir, sess.graph)
    
    model = ConvModel_270x480()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)
    whole_graph_def = sess.graph.as_graph_def()
    # fix whole_graph_def for bn
    for node in whole_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        whole_graph_def,  # The graph_def is used to retrieve the nodes
        ['output/roll']  # The output node names are used to select the usefull nodes
    )
    # output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def)
    # Finally we serialize and dump the output graph to the filesystem
    mode = "wb"
    with tf.gfile.GFile(args.output_graph, mode) as f:
        f.write(output_graph_def.SerializeToString())
        f.close()
    print("%d ops in the final graph." % len(output_graph_def.node))
    freeze_engine_from_pb(args)
    
if __name__ == '__main__':
    main()
