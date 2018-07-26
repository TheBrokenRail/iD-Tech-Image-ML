import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image

useCifar: bool

def getDataset():
    global useCifar
    print("Dataset:")
    print("  1. MNIST")
    print("  2. Cifar 10")
    option = input("> ")
    if option == "1":
        useCifar = False
    elif option == "2":
        useCifar = True
    else:
        print("Invalid Option!")
        getDataset()

getDataset()

if not useCifar:
    # Load MNIST data from tf examples
    image_height = 28
    image_width = 28

    color_channels = 1

    model_name = "mnist"

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    category_names = list(map(str, range(10)))

    print(train_data.shape)

    train_data = np.reshape(train_data, (-1, image_height, image_width, color_channels))

    print(train_data.shape)

    eval_data = np.reshape(eval_data, (-1, image_height, image_width, color_channels))
else:
    # Load cifar data from file
    image_height = 32
    image_width = 32

    color_channels = 3

    model_name = "cifar"

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = './cifar-10-data/'

    train_data = np.array([])
    train_labels = np.array([])

    def process_data(data):
        float_data = np.array(data, dtype=float) / 255.0

        reshaped_data = np.reshape(float_data, (-1, color_channels, image_height, image_width))

        transposed_data = np.transpose(reshaped_data, [0, 2, 3, 1])
        return transposed_data

    # Load all the data batches.
    for i in range(1,6):
        data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
        train_data = np.append(train_data, data_batch[b'data'])
        train_labels = np.append(train_labels, data_batch[b'labels'])

    # Load the eval batch.
    eval_batch = unpickle(cifar_path + 'test_batch')

    eval_data = eval_batch[b'data']
    eval_labels = eval_batch[b'labels']

    # Load the english category names.
    category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
    category_names = list(map(lambda x: x.decode("utf-8"), category_names_bytes))

    train_data = process_data(train_data)
    eval_data = process_data(eval_data)


class ConvNet:
    def __init__(self, image_height_proper, image_width_proper, channels, num_classes):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height_proper, image_width_proper, channels],
                                          name="inputs")
        print(self.input_layer.shape)
        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], padding="same",
                                        activation=tf.nn.relu)
        print(conv_layer_1.shape)

        pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)
        print(pooling_layer_1.shape)

        conv_layer_2 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], padding="same",
                                        activation=tf.nn.relu)
        print(conv_layer_2.shape)

        pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)
        print(pooling_layer_2.shape)

        flattened_pooling = tf.layers.flatten(pooling_layer_2)
        dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
        print(dense_layer.shape)

        dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.layers.dense(dropout, num_classes)
        print(outputs.shape)

        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)

        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)

        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())


training_steps = 20000
batch_size = 64

path = "./" + model_name + "-cnn/"

load_checkpoint = True
performance_graph = np.array([])
tf.reset_default_graph()

dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.shuffle(buffer_size=train_labels.shape[0])
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()

dataset_iterator = dataset.make_initializable_iterator()
next_element = dataset_iterator.get_next()

cnn = ConvNet(image_height, image_width, color_channels, 10)
saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    load_checkpoint = False
    os.makedirs(path)

trainNetwork: bool
evalNetwork: bool
testNetwork: bool
testCustomImage: bool

def getOptions():
    global trainNetwork, evalNetwork, testNetwork, testCustomImage
    print("Options:")
    print("  1. Train Network")
    print("  2. Evaluate Network")
    print("  3. Test Network")
    print("  4. Test Custom Image")
    print("  5. All")
    option = input("> ")
    if option == "1":
        trainNetwork = True
        evalNetwork = False
        testNetwork = False
        testCustomImage = False
    elif option == "2":
        trainNetwork = False
        evalNetwork = True
        testNetwork = False
        testCustomImage = False
    elif option == "3":
        trainNetwork = False
        evalNetwork = False
        testNetwork = True
        testCustomImage = False
    elif option == "4":
        trainNetwork = False
        evalNetwork = False
        testNetwork = False
        testCustomImage = True
    elif option == "5":
        trainNetwork = True
        evalNetwork = True
        testNetwork = True
        testCustomImage = True
    else:
        print("Invalid Option!")
        getOptions()

getOptions()

if trainNetwork:
    with tf.Session() as sess:
        if load_checkpoint:
            checkpoint = tf.train.get_checkpoint_state(path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        sess.run(tf.local_variables_initializer())
        sess.run(dataset_iterator.initializer)
        step: int
        for step in range(training_steps):
            current_batch = sess.run(next_element)

            batch_inputs = current_batch[0]
            batch_labels = current_batch[1]

            sess.run((cnn.train_operation, cnn.accuracy_op),
                     feed_dict={cnn.input_layer: batch_inputs, cnn.labels: batch_labels})
            sess.run((cnn.train_operation, cnn.accuracy_op),
                     feed_dict={cnn.input_layer: batch_inputs, cnn.labels: batch_labels})

            if step % 10 == 0:
                performance_graph = np.append(performance_graph, sess.run(cnn.accuracy))
            if step % 100 == 0 and step > 0:
                current_acc = sess.run(cnn.accuracy)

                print("Accuracy at step " + str(step) + ": " + str(current_acc * 100) + "%")
                print("Saving checkpoint")
                saver.save(sess, path + model_name, step, write_meta_graph=False)

        print("Saving final checkpoint for training session.")
        saver.save(sess, path + model_name, step, write_meta_graph=False)
        plt.plot(performance_graph)
        plt.figure().set_facecolor('white')
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.show()
if evalNetwork:
    with tf.Session() as sessEval:
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sessEval, checkpoint.model_checkpoint_path)
        sessEval.run(tf.local_variables_initializer())
        for image, label in zip(eval_data, eval_labels):
            sessEval.run(cnn.accuracy_op, feed_dict={cnn.input_layer: [image], cnn.labels: label})
        print("Accuracy: " + str(sessEval.run(cnn.accuracy) * 100) + "%")
if testNetwork:
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        indexes = np.random.choice(len(eval_data), 10, replace=False)

        rows = 5
        cols = 2

        fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
        fig.patch.set_facecolor('white')
        image_count = 0

        for idx in indexes:
            image_count += 1
            sub = plt.subplot(rows, cols, image_count)
            img = eval_data[idx]
            if model_name == "mnist":
                img = img.reshape(28, 28)
            plt.imshow(img)
            guess = sess.run(cnn.choice, feed_dict={cnn.input_layer: [eval_data[idx]]})
            if model_name == "mnist":
                guess_name = str(guess[0])
                actual_name = str(eval_labels[idx])
            else:
                guess_name = category_names[guess[0]]
                actual_name = category_names[eval_labels[idx]]
            sub.set_title("G: " + guess_name + " A: " + actual_name)
        plt.tight_layout()
        plt.show()
if testCustomImage:
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        sess.run(tf.local_variables_initializer())

        img = Image.open("Cat.png")
        if model_name == "mnist":
            img = img.resize((28, 328), Image.ANTIALIAS)
        else:
            img = img.resize((32, 32), Image.ANTIALIAS)
        img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
        guess = sess.run(cnn.choice, feed_dict={cnn.input_layer: [img]})
        if model_name == "mnist":
            guess_name = str(guess[0])
        else:
            guess_name = category_names[guess[0]]
        print("Guess: " + guess_name)
        plt.imshow(img)
        plt.show()
