#!/usr/bin/env python3

import random
import tables
import cv2

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 100
HIDDEN_LAYER_DEPTH = 1024

class steer_nn():
    """
    Neural network to predict steering wheel turning angle. 
    """

#    @profile
    def __init__(self):
        self.create_nn()
        self.create_tensorflow()

        self.input_data = np.zeros((BATCH_SIZE,90,320,3))
        self.angle_data = np.zeros((BATCH_SIZE))

        self.session = tf.InteractiveSession()

        # Merge all summaries and write them out
        self.merged_summaries = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter("./tmp",self.session.graph)
        self.test_writer = tf.train.SummaryWriter("./tmp")
        # Init session
        tf.initialize_all_variables().run()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, writh bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Convolutional NN impl.
    def create_nn(self):
        with tf.name_scope("input_layer"):
            # input layer - [batch size, 90 h, 320 w, 3 channels]
            self.img_in = tf.placeholder(tf.float32, [None, 90, 320, 3])

        # conv layers
        with tf.name_scope("conv_layer1"):
            Wc1 = self.weight_variable([5,5,3,24])
            bc1 = self.bias_variable([24])
            conv_layer1 = self.conv2d(self.img_in,Wc1,bc1,2)
            tf.image_summary("Convolution layer 1", tf.reshape(conv_layer1[0,:,:,:], [43,158,24,1]), max_images=100)

        with tf.name_scope("conv_layer2"):
            Wc2 = self.weight_variable([5,5,24,36])
            bc2 = self.bias_variable([36])
            conv_layer2 = self.conv2d(conv_layer1,Wc2,bc2,2)

        with tf.name_scope("conv_layer3"):
            Wc3 = self.weight_variable([5,5,36,48])
            bc3 = self.bias_variable([48])
            conv_layer3 = self.conv2d(conv_layer2,Wc3,bc3,2)

        with tf.name_scope("conv_layer4"):
            Wc4 = self.weight_variable([3,3,48,64])
            bc4 = self.bias_variable([64])
            conv_layer4 = self.conv2d(conv_layer3,Wc4,bc4,1)

        with tf.name_scope("conv_layer5"):
            Wc5 = self.weight_variable([3,3,64,64])
            bc5 = self.bias_variable([64])
            conv_layer5 = self.conv2d(conv_layer4,Wc5,bc5,1)

        with tf.name_scope("fully-conn_layer"):
            # Fully connected layer
            Wfc = self.weight_variable([4*33*64,HIDDEN_LAYER_DEPTH])
            bfc = self.bias_variable([HIDDEN_LAYER_DEPTH])
            conv_layer5_flat = tf.reshape(conv_layer5,[-1,4*33*64])
            fc_layer = tf.nn.relu(tf.matmul(conv_layer5_flat,Wfc) + bfc)

        with tf.name_scope("output_layer"):
            # Output  
            Wout = self.weight_variable([HIDDEN_LAYER_DEPTH,1])
            bout = self.bias_variable([1])
            self.predict_angle = tf.matmul(fc_layer,Wout) + bout

    def create_tensorflow(self):
        self.angle_truth = tf.placeholder(tf.float32, [None])
        self.cost = tf.reduce_mean(tf.square(self.angle_truth - self.predict_angle))
        # Monitor the cost of training
        tf.scalar_summary('Cost',self.cost)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

#    @profile
    def train(self):
        for batch_idx in range(len(self.train_idx)//BATCH_SIZE):
            self.input_ori = self.cam[self.train_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE], :, :, :]
            self.angle_data = self.angle[self.train_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]]
            print(batch_idx, self.input_ori.shape)
            print("train_idx's: ", self.train_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE])
            # Resize and normalize input image
            for img_cnt in range(self.input_ori.shape[0]):
                # Resize to x/2, y/2
                tmp_resized = np.uint8(cv2.resize(self.input_ori[img_cnt,:,:,:] ,(320, 90)))
                # Change to YUV colorspace
                tmp_yuv = cv2.cvtColor(tmp_resized,cv2.COLOR_BGR2YUV)
                # Normalization
                for channel in range(self.input_ori.shape[3]):
                    self.input_data[img_cnt,:,:,channel] = (tmp_yuv[:,:,channel]-tmp_yuv[:,:,channel].mean())/(tmp_yuv[:,:,channel].std()+1e-8)

            self.summary, _ = self.session.run([self.merged_summaries, self.optimizer], feed_dict={
                self.img_in:self.input_data,
                self.angle_truth:self.angle_data
            })

            # Record a summary for every batch
            self.test_writer.add_summary(self.summary,batch_idx)
        pass


    def test(self):
        pass

    def open_dataset(self, file):
        # Open HDF5 file 
        self.f = tables.open_file(file, 'r')

        # Training input data - camera image
        self.cam = self.f.root.center_camera
        # Labels - steering wheel angle
        self.angle = self.f.root.steering_angle
        # Speed is used to exclude invalid data (when speed = 0)
        self.speed = self.f.root.avg_speed

        # Shuffle data input, prepare for batch generation
        index = list(range(self.cam.shape[0]))
        #self.random.shuffle(index)
        # 70% train, 30% test
        self.train_idx, self.test_idx = train_test_split(index, test_size = 0.3)

    def close_dataset(self):
        # Close the dataset file
        self.f.close()

    def saveParm(self):
        # Save the scene
        save_path = self.saver.save(self.session, "./tmp/model_tr.ckpt")

def main():
    c2_net = steer_nn()

    c2_net.open_dataset('/home/vitob/ROS/dataset_1_center_camera_3ch.hdf5')
    c2_net.train()
    c2_net.close_dataset()

    c2_net.saveParm

    return


if __name__ == '__main__':
    main()
