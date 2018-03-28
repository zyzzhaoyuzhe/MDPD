import numpy as np

class Crowd_Sourcing_Readers():
    @staticmethod
    def read_data(file):
        """

        :param file:
        :return:
        """
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
