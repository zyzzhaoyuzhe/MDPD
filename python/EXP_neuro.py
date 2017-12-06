from __future__ import division

import os
import numpy as np
from MDPD import *
from PIL import Image
from Queue import Queue
import threading


data_bool = np.load('../neuroscience_dataset/fluordata1_bin.npy')
dim, ntime = data_bool.shape

# convert data to the format used by MDPD
data = np.zeros([dim, ntime, 2], dtype=np.int)
data[data_bool, 0] = 1
data[np.logical_not(data_bool), 1] = 1
data = np.transpose(data, (1, 0, 2))


def image_saver(queue):
    count = 0
    while True:
        img = queue.get()
        if img is None:
            break
        h = Image.fromarray(img.astype(np.uint8))
        h = h.resize((200, 200))
        h.save('../neuroscience_dataset/images/img_' + str(count).zfill(6) + '.png')
        count += 1

#
WINDOW = 500
STRIDE = 1
MAX = 0.5

images = Queue()

t = threading.Thread(target=image_saver, args=(images, ))
t.start()


for frame in xrange(0, ntime - WINDOW, STRIDE):
    img = utils.MI_data(data[frame:frame + WINDOW, :, :]).sum(axis=(1, 3))
    np.fill_diagonal(img, 0)
    # print img.max()
    # MAX = max(img.max(), MAX)
    img_uint8 = img / MAX * 255
    img_uint8 = np.clip(img_uint8, 0, 255)
    images.put(img_uint8)
images.put(None)
# print MAX
t.join()