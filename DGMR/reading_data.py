import pathlib
import tensorflow as tf
import functools
import random

seed = 17
tf.random.set_seed(seed)

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_sum': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'exists': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'date': tf.io.FixedLenFeature([], tf.string, default_value=''),
}


# todo: add normalization
def _parse_function(example_proto):
    try:
        example = tf.io.parse_single_example(example_proto, feature_description)
        example['image'] = tf.io.parse_tensor(example['image'], tf.float32, name=None)

        return example

    except tf.errors.InvalidArgumentError:
        raise

def only_image(image_dataset):
    return image_dataset['image']


# TODO add if date == first of month, if difference between dates bigger than 5 min?
def filter_windows(d, expected_len, ISS=True, ISS_value=200):
    window_ok = True
    if len(d['date']) != expected_len:
        window_ok = False
    for example in d['exists']:
        # the second column is whether the image is ok
        if example == 0:
            window_ok = False
            break
    # Importance Sampling Scheme
    # stochastically filter out sequences that contain little rainfall
    if ISS:
        rain_sum = 0.0
        for element in d['image_sum']:
            rain_sum += element
        prob = 1 - tf.math.exp(-(rain_sum / ISS_value))
        prob = tf.math.minimum(1.0, prob + 0.1)
        if prob < tf.random.uniform(shape=[]):
            window_ok = False
    return window_ok

def random_data(batch_size = 32):

    dataset = tf.data.Dataset.from_tensor_slices((tf.random.uniform([500, 22,256,256], maxval=20.0)))
    dataset = dataset.shuffle(buffer_size=32)  # TODO chnage
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def read_TFR(path, year=None, batch_size=32, window_size=22, window_shift=1):
    tfr_dir = pathlib.Path(path)
    if year == None:
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

    filter_function = functools.partial(filter_windows, expected_len=window_size)
    dataset = dataset.filter(filter_function)

    dataset = dataset.map(only_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

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
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=window_size, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=window_size, ISS_value=400)
    dataset = dataset.filter(filter_function)
    dataset = dataset.map(only_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

