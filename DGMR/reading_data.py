import pathlib
import tensorflow as tf
import functools

tf.random.set_seed(7)

feature_description = {
    'image':  tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_sum': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'exists': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'date': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

# todo: add normalization
def _parse_function(example_proto):

  example = tf.io.parse_single_example(example_proto, feature_description)
  example['image'] =tf.io.parse_tensor(example['image'], tf.float32, name=None)

  return example

def only_image(image_dataset):
  return image_dataset['image']

# TODO add if date == first of month, if difference between dates bigger than 5 min?
def filter_windows(d, expected_len, ISS= True):

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
        prob = 1 - tf.math.exp(-(rain_sum/200))
        prob = tf.math.minimum(1.0, prob + 0.1)
        if prob < tf.random.uniform(shape=[]):
                window_ok = False
    return window_ok



def read_TFR(path, year= None, batch_size = 32, window_size=22):

    tfr_dir = pathlib.Path(path)
    if year == None:
        files = list(tfr_dir.glob('*/*/*.tfrecords'))
    else:
        tfr_dir = pathlib.Path(path) / year
        files = list(tfr_dir.glob('*/*.tfrecords'))
    print("files",files)
    #files = tf.random.shuffle(files)
    raw_dataset = tf.data.TFRecordDataset(files)
    # dataset = shards.interleave(tf.data.TFRecordDataset)
    raw_dataset = raw_dataset.shuffle(buffer_size=32)
    parsed_dataset = raw_dataset.map(_parse_function)
    windows_dataset = parsed_dataset.window(size=window_size, shift=1)
    filter_function = functools.partial(filter_windows, expected_len=window_size)
    dataset = windows_dataset.filter(filter_function)
    dataset = dataset.map(only_image)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=64)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
