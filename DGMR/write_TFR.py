"""Write radar data into TFRecord files


 Args:
        1, 2, 3: year, month, day       of first radar file to be read
        4, 5, 6: year, month, day       of last radar file to be read.
        7: directory of radar data
        8: directory for TFR files
        example: 2009 1 1 2009 12 31 $HOME/radar_data $HOME/TFR_data

        Saves
        'image': image array  ,0s if data does not exist
        'image_sum': float sum of image
        'exists': boolean int if data exists
        'date': time stamp of image
         for every radar data file of a single day in a TFRecord file
        """


import pathlib
import pandas as pd
import h5py
import numpy as np
import tensorflow as tf
from pandas.tseries.offsets import DateOffset
import sys

start = pd.Timestamp(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
end = pd.Timestamp(int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
data_dir = pathlib.Path(sys.argv[7])
out_dir = pathlib.Path(sys.argv[8])

print(start)
print(end)
print(data_dir)
print(out_dir)

def check_h5_file(file):
    file_okay = 0
    try:
        f = h5py.File(file, 'r')
        image = f['image1']['image_data']
        image = np.asarray(image)
        x = image != 65535
        if np.any(x):
            file_okay = 1
        f.close()
    except:
        pass

    return file_okay

def prepare_rad_h5_file(file, upper_row=300, lowest_row=556, left_column=241, right_column=497 ):
    f = h5py.File(file, 'r')
    image = f['image1']['image_data']
    image = np.asarray(image, dtype=np.float32)
    f.close()
    image = image[upper_row:lowest_row, left_column:right_column]
    image = np.where( image == 65535, np.NaN, image / 100.0 )
    image_sum = np.sum(image)
    #print(image_sum)
    return image, image_sum

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _array_feature(array):
  serialized_array= tf.io.serialize_tensor(array)
  return _bytes_feature(serialized_array)

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def image_example(image, image_sum, status, date):
  feature = {
      'image': _array_feature(image),
      'image_sum': _float_feature(image_sum),
      'exists': _int64_feature(status),
      'date':_bytes_feature(date)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def write_day_TRfile(df, file_path, empty_image= tf.io.serialize_tensor(np.full([256, 256], np.nan))):

    if df.shape[0] != 12*24:
        raise AssertionError("Dataframe contains {r} instead of 12*24 rows".format(r=df.shape[0]))
    with tf.io.TFRecordWriter(file_path) as writer:
        for index in range(12*24):
            row = df.iloc[index, :]
            if bool(row['file_okay']):
                image, image_sum = prepare_rad_h5_file(row['file_observed'])
                example_message = image_example(image, image_sum, row['file_okay'],
                                                row['date'].encode())
            else:
                example_message = image_example(empty_image, 0, row['file_okay'],
                                                row['date'].encode())
            writer.write(example_message.SerializeToString())



def create_tfrecord_files(start,end,data_dir,out_dir):

    empty_image = tf.io.serialize_tensor(np.full([256, 256], np.nan))
    current_date = start
    while current_date <= end:
        print(current_date)
        following_date = current_date + DateOffset(1)
        year, month, day = current_date.strftime('%Y'), current_date.strftime('%m'), current_date.strftime('%d')
        file_names = list(data_dir.glob('{y}/{m}/*{y}{m}{d}*.h5'.format(y=year,m=month, d=day)))
        expected_dates = [dt.strftime('%Y%m%d%H%M') for dt in pd.date_range(start=current_date, end=following_date, freq="300s")][:-1]
        files_df = pd.DataFrame({'file': file_names})
        files_df['name'] = files_df['file'].apply(lambda path: path.name)
        expected_dates_df = pd.DataFrame({'date': expected_dates})
        expected_dates_df['file'] = expected_dates_df['date'].apply(
            lambda date: data_dir / date[4:6] / ('RAD_NL25_RAC_5min_' + date + '_cor.h5'))
        expected_dates_df['name'] = expected_dates_df['file'].apply(lambda path: path.name)
        merged_df = pd.merge(expected_dates_df, files_df, how='left', left_on='name', right_on='name',
                             suffixes=('_expected', '_observed'))
        print(pd.isna(merged_df['file_observed']).sum(), "files not found")
        merged_df['file_okay'] = merged_df['file_observed'].apply(check_h5_file)
        TFfile_path = out_dir / "{y}/{m}/".format(y=year,m=month)
        TFfile_path.mkdir(parents=True, exist_ok=True)
        shard_name = TFfile_path / "RAD_NL25_RAC_5min_{y}{m}{d}.tfrecords".format(y=year,m=month,d=day)
        write_day_TRfile(merged_df, str(shard_name), empty_image)
        current_date = following_date

create_tfrecord_files(start,end,data_dir,out_dir)

# TODO also calculate average and sd or max for normalizing
