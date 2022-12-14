import pathlib
import tensorflow as tf
import functools
import random

seed = 17
tf.random.set_seed(seed)

feature_description = {
    'image_radar': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_eth': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_sum_radar': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'exists_radar': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'exists_both': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'date': tf.io.FixedLenFeature([], tf.string, default_value=''),
}


# todo: add normalization
def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    example['image_radar'] = tf.io.parse_tensor(example['image_radar'], tf.float32, name=None)
    example['image_eth'] = tf.io.parse_tensor(example['image_eth'], tf.float32, name=None)

    return example




def only_image(image_dataset):

    return zip(image_dataset['image_eth'], image_dataset['image_radar'])

def only_image_test(image_dataset):

    return zip(image_dataset['date'], image_dataset['date'])


# TODO add if date == first of month, if difference between dates bigger than 5 min?
def filter_windows(d, expected_len, ISS=True, ISS_value=200):
    window_ok = True
    if len(d['date']) != expected_len:
        window_ok = False
    for example in d['exists_both']:
        # the second column is whether the image is ok
        if example == 0:
            window_ok = False
            break
    # Importance Sampling Scheme
    # stochastically filter out sequences that contain little rainfall
    if ISS:
        rain_sum = 0.0
        for element in d['image_sum_radar']:
            rain_sum += element
        prob = 1 - tf.math.exp(-(rain_sum / ISS_value))
        prob = tf.math.minimum(1.0, prob + 0.1)
        if prob < tf.random.uniform(shape=[]):
            window_ok = False
    return window_ok


def read_TFR(path, year=None, batch_size=32, window_size=22, window_shift=1):
    tfr_dir = pathlib.Path(path)
    if year == None:
        pattern = str(tfr_dir / '*/*/*.tfrecords')

    else:
        tfr_dir = pathlib.Path(path) / str(year)
        pattern = str(tfr_dir) + '/*/*.tfrecords'
    print(pattern)
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(pattern, seed=seed))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    ## dataset = shards.interleave(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=window_size, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=window_size,  ISS_value=200)
    dataset = dataset.filter(filter_function)
    dataset = dataset.map(only_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.shuffle(buffer_size=100)  # TODO chnage
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def read_TFR_test(path, year=None, batch_size=32, window_size=22, window_shift=22):
    tfr_dir = pathlib.Path(path)
    if year == None:
        print("all years")
        pattern = str(tfr_dir / '*/*/*.tfrecords')
    else:
        tfr_dir = pathlib.Path(path) / str(year)
        pattern = str(tfr_dir) + '/*/*.tfrecords'
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(pattern, seed=seed))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    ## dataset = shards.interleave(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=window_size, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=window_size, ISS_value=400)
    dataset = dataset.filter(filter_function)
    dataset = dataset.map(only_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def check_TFR(path, year=None, batch_size=32, window_size=22, window_shift=1):
    tfr_dir = pathlib.Path(path)
    if year == None:
        pattern = str(tfr_dir / '*/*/*.tfrecords')

    else:
        tfr_dir = pathlib.Path(path) / str(year)
        pattern = str(tfr_dir) + '/*/*.tfrecords'
    print(pattern)
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(pattern, seed=seed))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    ## dataset = shards.interleave(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=window_size, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=window_size,  ISS_value=0)
    dataset = dataset.filter(filter_function)
    dataset = dataset.map(only_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    #dataset = dataset.shuffle(buffer_size=100)  # TODO chnage
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


y = check_TFR("/Users/frederikesmac/MA/Data/TFR_ETH", batch_size=1)
print(y)
for i in y:
    print(i)
    print("-----------------")

    #print(i['date'])
    #break

