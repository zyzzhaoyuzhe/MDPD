import configparser
import sys
import os

from MDPD import utils, readers, MDPD

folder = "/media/vzhao/Data/MNIST"
# folder = "/Users/vincent/Documents/Research/MDPD/MNIST"
mnist = readers.MNIST_Reader(folder, binarized=True)
train, labels = mnist.train, mnist.labels
_, DIM, _ = train.shape

# data per digit
train_uni = [None] * 10
for dig in range(10):
    train_uni[dig] = train[labels==dig,...]
# small sample
train_small = train[:20000,...]
labels_small = labels[:20000]

if __name__ == '__main__':
    exp_folder = os.path.abspath(sys.argv[1])

    config = configparser.RawConfigParser()
    config.read(os.path.join(exp_folder, 'configuration.cfg'))

    nsample = config.getint('MNIST_config', 'nsample')
    depth = config.getint('MNIST_config', 'depth')
    width = config.getint('MNIST_config', 'width')
    features = config.getint('MNIST_config', 'features')
    epoch = config.getint('MNIST_config', 'epoch')
    batch = config.getint('MNIST_config', 'batch')


    data, labs = train[:nsample,...], labels[:nsample]

    hie_model = MDPD.Hierachical_MDPD(depth=depth)
    hie_model.fit(data, features, epoch=epoch, batch=batch)
    hie_model.save(os.path.join(exp_folder, 'model.p'))