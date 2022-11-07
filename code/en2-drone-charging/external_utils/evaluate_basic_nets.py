import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
ms = ['/root/redflags-honza/external_utils/models/(9, 8, 7)', '/root/redflags-honza/external_utils/models/x9842', '/root/redflags-honza/external_utils/models/2022_03_30_03-33-46-0.9779280424118042.h5', '/root/redflags-honza/external_utils/models/2022_03_30_03-11-01-0.9774134755134583.h5', '/root/redflags-honza/external_utils/models/2022_03_30_00-24-19-0.9771632552146912.h5', '/root/redflags-honza/external_utils/models/2022_03_29_23-23-11-0.9767979383468628.h5', '/root/redflags-honza/external_utils/models/2022_03_30_13-54-40-0.9765704870223999.h5', '/root/redflags-honza/external_utils/models/2022_03_30_00-50-41-0.9765072464942932.h5', '/root/redflags-honza/external_utils/models/2022_03_30_01-20-38-0.9763468503952026.h5', '/root/redflags-honza/external_utils/models/2022_03_30_12-53-09-0.9763191938400269.h5', '/root/redflags-honza/external_utils/models/2022_03_31_12-36-53-0.9762064814567566.h5', '/root/redflags-honza/external_utils/models/2022_03_31_09-48-32-0.9761964678764343.h5']

for file in ['/dev/shm/datasets/validation/16667_AIO.npz', '/dev/shm/datasets/testing/16667_AIO.npz']:
    d = np.load(file, mmap_mode='r')
    X, y = d['data'], d['labels']

    res = [(tf.keras.models.load_model(m, custom_objects={'StochasticDepth': tfa.layers.StochasticDepth}).evaluate(X, y, batch_size=3000), m) for m in ms]
    print(res)
