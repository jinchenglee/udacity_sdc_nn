#!/usr/bin/env python3

import random
import tables
import cv2

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
HIDDEN_LAYER_DEPTH = 1024

class steer_nn():
    """
    Neural network to predict steering wheel turning angle. 
    """

#    @profile
    def __init__(self):
        self.create_nn()
        self.create_tensorflow()

        self.input_data = np.zeros((BATCH_SIZE,45,160,3))
        self.angle_data = np.zeros((BATCH_SIZE))
        self.scaled_angle_data = np.zeros((BATCH_SIZE))

        self.session = tf.InteractiveSession()

        # Merge all summaries and write them out
        self.merged_summaries = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter("./tmp",self.session.graph)
        self.test_writer = tf.train.SummaryWriter("./tmp")
        # Init session
        tf.initialize_all_variables().run()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def batch_norm(self, x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
    
            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def weight_variable(self,shape,stddev=0.2):
        initial = tf.truncated_normal(shape,stddev=stddev)
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
            # input layer - [batch size, 45 h, 160 w, 3 channels]
            self.img_in = tf.placeholder(tf.float32, [None, 45, 160, 3])
            tf.histogram_summary("input_img_in", self.img_in);

        # conv layers
        with tf.name_scope("conv_layer1"):
            Wc1 = self.weight_variable([5,5,3,24])
            bc1 = self.bias_variable([24])
            conv_layer1 = self.conv2d(self.img_in,Wc1,bc1,2)
            tf.image_summary("Convolution layer 1", tf.reshape(conv_layer1[0,:,:,:], [21,78,24,1]), max_images=100)
            tf.histogram_summary("Wc1", Wc1);
            tf.histogram_summary("bc1", bc1);
            #tf.scalar_summary('conv_1',conv_layer1)

        with tf.name_scope("conv_layer2"):
            Wc2 = self.weight_variable([5,5,24,36])
            bc2 = self.bias_variable([36])
            conv_layer2 = self.conv2d(conv_layer1,Wc2,bc2,2)
            tf.histogram_summary("Wc2", Wc2);
            tf.histogram_summary("bc2", bc2);
            #tf.scalar_summary('conv_2',conv_layer2)

        with tf.name_scope("conv_layer3"):
            Wc3 = self.weight_variable([5,5,36,48])
            bc3 = self.bias_variable([48])
            conv_layer3 = self.conv2d(conv_layer2,Wc3,bc3,2)
            tf.histogram_summary("Wc3", Wc3);
            tf.histogram_summary("bc3", bc3);
            #tf.scalar_summary('conv_3',conv_layer3)

        with tf.name_scope("conv_layer4"):
            Wc4 = self.weight_variable([3,3,48,64])
            bc4 = self.bias_variable([64])
            conv_layer4 = self.conv2d(conv_layer3,Wc4,bc4,1)
            tf.histogram_summary("Wc4", Wc4);
            tf.histogram_summary("bc4", bc4);
            #tf.scalar_summary('conv_4',conv_layer4)

        #with tf.name_scope("conv_layer5"):
        #    Wc5 = self.weight_variable([3,3,64,64])
        #    bc5 = self.bias_variable([64])
        #    conv_layer5 = self.conv2d(conv_layer4,Wc5,bc5,1)

        with tf.name_scope("fully-conn_layer"):
            # Fully connected layer
            Wfc = self.weight_variable([1*15*64,HIDDEN_LAYER_DEPTH])
            bfc = self.bias_variable([HIDDEN_LAYER_DEPTH])
            conv_layer5_flat = tf.reshape(conv_layer4,[-1,1*15*64])
            fc_layer = tf.nn.relu(tf.matmul(conv_layer5_flat,Wfc) + bfc)
            tf.histogram_summary("Wfc", Wfc);
            tf.histogram_summary("bfc", bfc);
            #tf.scalar_summary('conv_5_flat',conv_layer5_flat)

        with tf.name_scope("output_layer"):
            # Output  
            Wout = self.weight_variable([HIDDEN_LAYER_DEPTH,1])
            bout = self.bias_variable([1])
            self.predict_angle = tf.matmul(fc_layer,Wout) + bout
            tf.histogram_summary("Wout", Wout);
            tf.histogram_summary("bout", bout);
            tf.histogram_summary('pred_angle_hist',self.predict_angle)

    def create_tensorflow(self):
        self.angle_truth = tf.placeholder(tf.float32, [None])
        self.cost = tf.reduce_mean(tf.square(self.angle_truth - self.predict_angle))
        # Monitor the cost of training
        tf.scalar_summary('Cost',self.cost)
        self.optimizer = tf.train.AdamOptimizer(0.002).minimize(self.cost)

#    @profile
    def train(self):
        # Reshuffle the training set
        random.shuffle(self.train_idx)
        for batch_idx in range(len(self.train_idx)//BATCH_SIZE):
            self.input_ori = self.cam[self.train_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE], :, :, :]
            self.angle_data = self.angle[self.train_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]]
            self.speed_data = self.speed[self.train_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]]
            print(batch_idx, self.input_ori.shape)
            #print("train_idx's: ", self.train_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE])
            # Resize and normalize input image
            local_count = 0
            for img_cnt in range(self.input_ori.shape[0]):
                if self.speed_data[img_cnt] != 0 :
                    # Resize to x/2, y/2
                    tmp_resized = np.uint8(cv2.resize(self.input_ori[img_cnt,:,:,:] ,(160, 45)))
                    # Change to YUV colorspace
                    #tmp_yuv = cv2.cvtColor(tmp_resized,cv2.COLOR_BGR2YUV)
                    # Normalization
                    for channel in range(self.input_ori.shape[3]):
                        #self.input_data[local_count,:,:,channel] = (tmp_resized[:,:,channel]-tmp_resized[:,:,channel].mean())/tmp_resized[:,:,channel].std()
                        self.input_data[local_count,:,:,channel] = (tmp_resized[:,:,channel]-tmp_resized[:,:,channel].mean())/(np.max(tmp_resized[:,:,channel])-np.min(tmp_resized[:,:,channel]))
                    self.scaled_angle_data[local_count] = self.angle[img_cnt]*100
                    local_count +=1

            print("local_count = ", local_count)

            self.summary, _ = self.session.run([self.merged_summaries, self.optimizer], feed_dict={
                self.img_in:self.input_data,
                self.angle_truth:self.scaled_angle_data
            })

            # Record a summary for every batch
            self.test_writer.add_summary(self.summary,batch_idx)

    def test(self):
        for batch_idx in range(1):
            self.input_ori = self.cam[self.test_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE], :, :, :]
            self.angle_data = self.angle[self.test_idx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]]

            for img_cnt in range(self.input_ori.shape[0]):
                # Resize to x/2, y/2
                tmp_resized = np.uint8(cv2.resize(self.input_ori[img_cnt,:,:,:] ,(160, 45)))
                # Change to YUV colorspace
                #tmp_yuv = cv2.cvtColor(tmp_resized,cv2.COLOR_BGR2YUV)
                # Save image
                cv2.imwrite("./tmp/test_"+str(img_cnt)+".png",tmp_resized)
                # Normalization
                for channel in range(self.input_ori.shape[3]):
                    self.input_data[img_cnt,:,:,channel] = (tmp_resized[:,:,channel]-tmp_resized[:,:,channel].mean())/(np.max(tmp_resized[:,:,channel])-np.min(tmp_resized[:,:,channel]))

            pred_angle_eval = self.predict_angle.eval(feed_dict = 
                    { 
                        self.img_in     :   self.input_data
                    })
            loss = tf.reduce_mean(tf.square(self.angle_data*100 - pred_angle_eval))
            test_out = np.zeros((len(self.angle_data),4))
            # index
            test_out[:,0] = range(len(self.angle_data))
            # Ground truth
            test_out[:,1] = self.angle_data*100
            # Predicted value
            test_out[:,2] = pred_angle_eval[:,0]
            # Delta
            test_out[:,3] = test_out[:,1] - test_out[:,2]
            #print("Ground truth: ", self.angle_data*100)
            #print("Prediction: ", pred_angle_eval)
            print(test_out)
            print('Test batch: ', batch_idx, ' Accuracy = ', loss.eval() )

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
        # 70% train, 30% test
        self.train_idx, self.test_idx = train_test_split(index, test_size = 0.2)

    def close_dataset(self):
        # Close the dataset file
        self.f.close()

    def saveParm(self):
        # Save the scene
        print("Saving parameters...")
        save_path = self.saver.save(self.session, "./tmp/model_tr.ckpt")

    def restoreParam(self):
        # Restore the scene
        self.saver.restore(self.session, "./record/model_tr_1.ckpt")
        pass


def main():
    c2_net = steer_nn()

    c2_net.open_dataset('/home/vitob/ROS/dataset_1_center_camera_3ch.hdf5')

    # Training
    for epoch in range(1):
        c2_net.train()
        c2_net.test()
    # Evaluation
    #c2_net.restoreParam()
    #c2_net.test()

    c2_net.close_dataset()

    c2_net.saveParm()

    return


if __name__ == '__main__':
    main()
