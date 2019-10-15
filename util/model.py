import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init


class CPMNets():
    """build model
    """
    def __init__(self, view_num, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        # initialize parameter
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        # initialize latent space data
        self.h_train, self.h_train_update = self.H_init('train')
        self.h_test, self.h_test_update = self.H_init('test')
        self.h = tf.concat([self.h_train, self.h_test], axis=0)
        self.h_index = tf.placeholder(tf.int32, shape=[None, 1], name='h_index')
        self.h_temp = tf.gather_nd(self.h, self.h_index)
        # initialize the input data
        self.input = dict()
        self.sn = dict()
        for v_num in range(self.view_num):
            self.input[str(v_num)] = tf.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]],
                                                    name='input' + str(v_num))
            self.sn[str(v_num)] = tf.placeholder(tf.float32, shape=[None, 1], name='sn' + str(v_num))
        # ground truth
        self.gt = tf.placeholder(tf.int32, shape=[None], name='gt')
        # bulid the model
        self.train_op, self.loss = self.bulid_model([self.h_train_update, self.h_test_update], learning_rate)
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def bulid_model(self, h_update, learning_rate):
        # initialize network
        net = dict()
        for v_num in range(self.view_num):
            net[str(v_num)] = self.Encoding_net(self.h_temp, v_num)
        # calculate reconstruction loss
        reco_loss = self.reconstruction_loss(net)
        # calculate classification loss
        class_loss = self.classification_loss()
        all_loss = tf.add(reco_loss, self.lamb * class_loss)
        # train net operator
        # train the network to minimize reconstruction loss
        train_net_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=tf.get_collection('weight'))
        # train the latent space data to minimize reconstruction loss and classification loss
        train_hn_op = tf.train.AdamOptimizer(learning_rate[1]) \
            .minimize(all_loss, var_list=h_update[0])
        # adjust the latent space data
        adj_hn_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=h_update[1])
        return [train_net_op, train_hn_op, adj_hn_op], [reco_loss, class_loss, all_loss]

    def H_init(self, a):
        with tf.variable_scope('H' + a):
            if a == 'train':
                h = tf.Variable(xavier_init(self.trainLen, self.lsd_dim))
            elif a == 'test':
                h = tf.Variable(xavier_init(self.testLen, self.lsd_dim))
            h_update = tf.trainable_variables(scope='H' + a)
        return h, h_update

    def Encoding_net(self, h, v):
        weight = self.initialize_weight(self.layer_size[v])
        layer = tf.matmul(h, weight['w0']) + weight['b0']
        for num in range(1, len(self.layer_size[v])):
            layer = tf.nn.dropout(tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)], 0.9)
        return layer

    def initialize_weight(self, dims_net):
        all_weight = dict()
        with tf.variable_scope('weight'):
            all_weight['w0'] = tf.Variable(xavier_init(self.lsd_dim, dims_net[0]))
            all_weight['b0'] = tf.Variable(tf.zeros([dims_net[0]]))
            tf.add_to_collection("weight", all_weight['w' + str(0)])
            tf.add_to_collection("weight", all_weight['b' + str(0)])
            for num in range(1, len(dims_net)):
                all_weight['w' + str(num)] = tf.Variable(xavier_init(dims_net[num - 1], dims_net[num]))
                all_weight['b' + str(num)] = tf.Variable(tf.zeros([dims_net[num]]))
                tf.add_to_collection("weight", all_weight['w' + str(num)])
                tf.add_to_collection("weight", all_weight['b' + str(num)])
        return all_weight

    def reconstruction_loss(self, net):
        loss = 0
        for num in range(self.view_num):
            loss = loss + tf.reduce_sum(
                tf.pow(tf.subtract(net[str(num)], self.input[str(num)])
                       , 2.0) * self.sn[str(num)]
            )
        return loss

    def classification_loss(self):
        F_h_h = tf.matmul(self.h_temp, tf.transpose(self.h_temp))
        F_hn_hn = tf.diag_part(F_h_h)
        F_h_h = tf.subtract(F_h_h, tf.matrix_diag(F_hn_hn))
        classes = tf.reduce_max(self.gt) - tf.reduce_min(self.gt) + 1
        label_onehot = tf.one_hot(self.gt - 1, classes)  # gt begin from 1
        label_num = tf.reduce_sum(label_onehot, 0, keep_dims=True)  # should sub 1.Avoid numerical errors
        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        label_num_broadcast = tf.tile(label_num, [self.trainLen, 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)
        gt_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.reduce_max(F_h_h_mean, axis=1, keep_dims=False)
        theta = tf.cast(tf.not_equal(self.gt, gt_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot)
        F_h_hn_mean = tf.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean')
        return tf.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))

    def train(self, data, sn, gt, epoch, step=[5, 5]):
        global Reconstruction_LOSS
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt[index]
        sn = sn[index]
        feed_dict = {self.input[str(v_num)]: data[str(v_num)][index] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.trainLen, 1) for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})
        for iter in range(epoch):
            # update the network
            for i in range(step[0]):
                _, Reconstruction_LOSS, Classification_LOSS = self.sess.run(
                    [self.train_op[0], self.loss[0], self.loss[1]], feed_dict=feed_dict)
            # update the h
            for i in range(step[1]):
                _, Reconstruction_LOSS, Classification_LOSS = self.sess.run(
                    [self.train_op[1], self.loss[0], self.loss[1]], feed_dict=feed_dict)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, Classification Loss = {:.4f} " \
                .format((iter + 1), Reconstruction_LOSS, Classification_LOSS)
            print(output)

    def test(self, data, sn, gt, epoch):
        feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.testLen, 1) for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index:
                              np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen})
        for iter in range(epoch):
            # update the h
            for i in range(5):
                _, Reconstruction_LOSS = self.sess.run(
                    [self.train_op[2], self.loss[0]], feed_dict=feed_dict)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}" \
                .format((iter + 1), Reconstruction_LOSS)
            print(output)

    def get_h_train(self):
        lsd = self.sess.run(self.h)
        return lsd[0:self.trainLen]

    def get_h_test(self):
        lsd = self.sess.run(self.h)
        return lsd[self.trainLen:]
