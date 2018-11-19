import tensorflow
import numpy
import os
from matplotlib import pyplot


class ConvNN(object):

    def __init__(self, batch_size=None, epochs=None, learning_rate=None, random_seed=None,
                 c1_kernel_size=None, p1_pool_size=None, c2_kernel_size=None, p2_pool_size=None, c1_op_channel=None,
                 c2_op_channel=None):
        numpy.random.seed(random_seed)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.c1_kernel_size = c1_kernel_size
        self.c2_kernel_size = c2_kernel_size
        self.c1_op_channel = c1_op_channel
        self.c2_op_channel = c2_op_channel
        self.p1_pool_size = p1_pool_size
        self.p2_pool_size = p2_pool_size

        g = tensorflow.Graph()
        with g.as_default():
            tensorflow.set_random_seed(random_seed)
            self.build()
            self.init_var = tensorflow.global_variables_initializer()
            self.saver = tensorflow.train.Saver()

        self.session = tensorflow.Session(graph=g)

    def build(self):
        tfx = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 784], name='tfx')

        tfy = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='tfy')

        tfx_image = tensorflow.reshape(tensor=tfx, shape=[-1, 28, 28, 1], name='tfx_2d_image')

        tfy_one_hot = tensorflow.one_hot(indices=tfy, depth=10, dtype=tensorflow.float32, name='tfy_one_hot')

        conv1 = tensorflow.layers.conv2d(tfx_image, kernel_size=(self.c1_kernel_size, self.c1_kernel_size),
                                         filters=self.c1_op_channel, activation=tensorflow.nn.relu)

        pool1 = tensorflow.layers.max_pooling2d(conv1, pool_size=(self.p1_pool_size, self.p1_pool_size), strides=(2, 2))

        conv2 = tensorflow.layers.conv2d(pool1, kernel_size=(self.c2_kernel_size, self.c2_kernel_size), strides=(3, 3),
                                         filters=self.c2_op_channel, activation=tensorflow.nn.relu)

        pool2 = tensorflow.layers.max_pooling2d(conv2, pool_size=(self.p2_pool_size, self.p2_pool_size), strides=(4, 4))

        ip_shape = pool2.get_shape().as_list()
        ip_units = numpy.prod(ip_shape[1:])
        pool2_flat = tensorflow.reshape(pool2, shape=[-1, ip_units])
        fconn1 = tensorflow.layers.dense(inputs=pool2_flat, units=10, activation=None)

        predictions = {
            'probabilities': tensorflow.nn.softmax(fconn1, name='probabilities'),
            'labels': tensorflow.argmax(input=fconn1, axis=1, name='labels')
        }

        cross_ent_loss = tensorflow.reduce_mean(
            input_tensor=tensorflow.nn.softmax_cross_entropy_with_logits_v2(logits=fconn1, labels=tfy_one_hot),
            name='cross_ent_loss')

        optim = tensorflow.train.AdamOptimizer(self.learning_rate)
        optim = optim.minimize(loss=cross_ent_loss, name='train_optim')

        cor_preds = tensorflow.equal(predictions['labels'], tensorflow.cast(x=tfy, dtype=tensorflow.int64),
                                     name='correct_predictions')
        accuracy = tensorflow.reduce_mean(tensorflow.cast(x=cor_preds, dtype=tensorflow.float32), name='accuracy')

    def train(self, X_train, Y_train, X_validate=None, Y_validate=None, initialize=True):
        if initialize:
            self.session.run(self.init_var)

        loss_grad = []
        for ep in range(1, self.epochs+1):
            avg_loss = 0
            bgen = self.batch_gen(X_train, Y_train, batch_size=self.batch_size)
            for i, (batch_x, batch_y) in enumerate(bgen):
                feed = {'tfx:0': batch_x, 'tfy:0': batch_y}
                loss, _ = self.session.run(['cross_ent_loss:0', 'train_optim'], feed_dict=feed)
                avg_loss = avg_loss + loss
            loss_grad.append(avg_loss)
            print('Epoch % 02d: Training Avg.Loss: ' '% 7.3f' % (ep, avg_loss), end=' ')

            if not (X_validate is None or Y_validate is None):
                feed = {'tfx:0': X_validate, 'tfy:0': Y_validate}
                vl_acc = self.session.run('accuracy:0', feed_dict=feed)
                print('Validation accuracy: % 7.3f' % vl_acc)
            else:
                print()

        fig = pyplot.figure(figsize=(13, 9), dpi=200, facecolor='w', edgecolor='k')
        pyplot.plot(range(0, self.epochs), loss_grad, "-o")
        pyplot.title("CNN : MNIST \nCN1:" + self.c1_kernel_size + "X" + self.c1_kernel_size + "X" + self.c1_op_channel +
                     " P1:" + self.p1_pool_size + "X" + self.p1_pool_size + " C2:" + self.c2_kernel_size + "X" +
                     self.c2_kernel_size + "X" + self.c2_op_channel + " P2:" + self.p2_pool_size + "X" +
                     self.p2_pool_size + " FC: 10")
        pyplot.xlabel("Epoch")
        pyplot.ylabel("average-loss")
        fig.savefig('MNIST-loss-decrease-CNN.png')

    def predict(self, X_test, type='labels'):
        feed = {'tfx:0': X_test}
        if type is 'probabilities':
            pred = self.session.run('probabilities:0', feed_dict=feed)
        else:
            pred = self.session.run('labels:0', feed_dict=feed)
        return pred

    def batch_gen(self, X, Y, batch_size, shuffle=True, random_seed=1):
        all_idn = numpy.arange(Y.shape[0])
        if shuffle:
            rnd_dev = numpy.random.RandomState(random_seed)
            rnd_dev.shuffle(all_idn)
            X = X[all_idn, :]
            Y = Y[all_idn]
        for i in range(0, X.shape[0], batch_size):
            yield (X[i:(i+batch_size),:], Y[i:(i+batch_size)])

    def save(self, epoch, path='./tf-layer-models/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.saver.save(sess=self.session, save_path=os.path.join(path, 'model.ckpt'), global_step=epoch)

    def load(self, epoch, path):
        self.saver.restore(self.session, os.path.join(path, 'model.ckpt-%d' % epoch))