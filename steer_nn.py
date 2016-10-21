#!/usr/bin/env python3

import random
import tables

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 200

class steer_nn():
    """
    Neural network to predict steering wheel turning angle. 
    """

    def __init__(self):
        self.create_nn()
        self.create_tensorflow()

        #self.session = tf.InteracriveSession()

    def create_nn(self):
        pass

    def create_tensorflow(self):
        pass

    def train(self):
        for batch_idx in range(len(self.train_idx)//BATCH_SIZE):
            self.input_data = self.cam[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE, :, :, :]
            print(batch_idx, self.input_data.shape)
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
        self.f.close

def main():
    c2_net = steer_nn()

    c2_net.open_dataset('/home/vitob/ROS/dataset_1_center_camera_3ch.hdf5')
    c2_net.train()
    c2_net.close_dataset()

    return


if __name__ == '__main__':
    main()
