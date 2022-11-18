import pathlib
import tensorflow as tf
import functools
import random
import h5py
import numpy as np

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

def prepare_eth_h5_file(file, upper_row=300, lowest_row=556, left_column=241, right_column=497 ):
    f = h5py.File(file, 'r')
    image = f['image1']['image_data']
    image = np.asarray(image, dtype=np.float32)
    f.close()
    image = image[upper_row:lowest_row, left_column:right_column]
     # out of image == 255
     # changed from NaN to 0
    image = np.where(image == 255, 0.0, image * 0.062992)
    # missing value == 0 TODO check if this needs to stay 0
    image = np.where(image == 0, -1.0, image)
    if (np.isnan(image).any()):
        print(file, "contains nan values in ETH")

    #print(image_sum)
    return image


def prepare_radar_h5_file(file, upper_row=300, lowest_row=556, left_column=241, right_column=497 ):
    f = h5py.File(file, 'r')
    image = f['image1']['image_data']
    image = np.asarray(image, dtype=np.float32)
    f.close()
    image = image[upper_row:lowest_row, left_column:right_column]
    image = np.where( image == 65535, np.NaN, image / 100.0 )

    image_sum = np.sum(image)

    if (np.isnan(image).any()):
        print(file, "contains nan values in radar")

    #print(image_sum)
    return image, image_sum

# todo: add normalization
def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example
    if example['exists_both'] == 1:
        pr = example['image_radar'].numpy().decode("utf-8")
        example['image_radar'] = prepare_radar_h5_file(pr)
       # example['image_eth'] = prepare_radar_h5_file(example['image_eth'].numpy().decode("utf-8"))
    # if there is no valid eth image
    elif example['exists_radar'] == 1:
        example['image_radar'] = prepare_radar_h5_file(example['image_radar'].numpy().decode("utf-8"))
        example['image_eth'] = np.full([1, 1], 0, dtype=np.float32)
    # if there are no valid radar and eth images
    else:
        example['image_radar'] = np.full([1, 1], 0, dtype=np.float32)
        example['image_eth'] = np.full([1, 1], 0, dtype=np.float32)

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
    return dataset

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
print("HHHHYYY")
print(y)

for i in y:
    print(i)
    print(i['image_radar'])
    print(i['image_radar'].decode("utf-8"))

    print(i['image_radar'].numpy().decode("utf-8"))
    print(prepare_radar_h5_file(i['image_radar'].numpy().decode("utf-8")))
    print("-----------------")
    break

    #print(i['date'])
    #break



# dont turn into arrays yet
# first filter
# then use parse again
# doesn't matter too mich