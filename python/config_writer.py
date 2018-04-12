import configparser

config = configparser.RawConfigParser()

config.add_section('MNIST_config')

config.set('MNIST_config', 'nsample', '20000')

config.set('MNIST_config', 'model', 'standard')

config.set('MNIST_config', 'depth', '3')
config.set('MNIST_config', 'width', '2')
config.set('MNIST_config', 'features', '300')
config.set('MNIST_config', 'epoch', '100')

with open('/home/vzhao/Documents/Projects/MDPD/results/MNIST_hmdpd_depth_6.cfg', 'wb') as configfile:
    config.write(configfile)
