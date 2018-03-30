import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(10)

class Crowd_Sourcing_Readers():
    @staticmethod
    def read_data(file):
        """Read crowdsourcing data"""
        with open(file, 'r') as h:
            cache = []
            j_max, i_max, label_max = 0, 0, 0
            for line in h:
                j, i, label = map(lambda x: int(x) - 1, line.strip().split('\t'))
                j_max = max(j_max, j + 1)
                i_max = max(i_max, i + 1)
                label_max = max(label_max, label + 1)
                cache.append((j, i, label))
        output = np.zeros([j_max, i_max, label_max], dtype=np.int)
        for j, i, label in cache:
            output[j, i, label] = 1
        if np.any(output.sum(axis=2) > 1):
            raise 'Data Error'
        if np.any(output.sum(axis=2) == 0):
            logger.info('Data has missing values. A new label is created to represent the missing values.')
            output = np.concatenate((output, 1 - output.sum(axis=2, keepdims=True)), axis=2)
        logger.info('The Data has {} dimensions, {} samples, and {} volcabulary size.'.format(output.shape[1], output.shape[0], output.shape[2]))
        return output

    @staticmethod
    def read_label(file):
        with open(file, 'r') as h:
            cache = []
            j_max, l_max = 0, 0
            for line in h:
                j, l = map(lambda x: int(x) - 1, line.strip().split('\t'))
                j_max = max(j_max, j + 1)
                l_max = max(l_max, l + 1)
                cache.append((j, l))
            output = np.zeros(j_max, dtype=np.int)
            for j, l in cache:
                output[j] = l
        return output
