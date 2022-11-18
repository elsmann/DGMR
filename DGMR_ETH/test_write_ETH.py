"""Write ETH data into TFRecord files


 Args:
        1, 2, 3: year, month, day       of first radar file to be read
        4, 5, 6: year, month, day       of last radar file to be read.
        7: directory of radar data
        8: directory of ETH data
        9: directory for TFR files
        example: 2009 1 1 2009 12 31 $HOME/radar_data $HOME/ETH_data $HOME/TFR_data

        Saves
        'image': image array  ,0s if data does not exist
        'image_sum': float sum of image
        'exists': boolean int if data exists
        'date': time stamp of image
         for every eth data file of a single day in a TFRecord file
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
data_dir_radar = pathlib.Path(sys.argv[7])
data_dir_eth = pathlib.Path(sys.argv[8])
out_dir = pathlib.Path(sys.argv[9])

print(start)
print(end)
print(data_dir_radar)
print(data_dir_eth)

print(out_dir)

def check_eth_h5_file(file):
    file_okay = 0
    try:
        f = h5py.File(file, 'r')
        image = f['image1']['image_data']
        image = np.asarray(image)
        x = image != 255
        if np.any(x):
            file_okay = 1
        f.close()
    except:
        pass

    return file_okay

def check_radar_h5_file(file):
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

def prepare_eth_h5_file(file, upper_row=300, lowest_row=556, left_column=241, right_column=497 ):
    f = h5py.File(file, 'r')
    image = f['image1']['image_data']
    image = np.asarray(image, dtype=np.float32)
    f.close()
    image = image[upper_row:lowest_row, left_column:right_column]
     # out of image == 255
    image = np.where(image == 255, np.NaN, image * 0.062992)
    # missing value == 0 TODO check if this needs to stay 0
    image = np.where(image == 0, np.NaN, image)
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

def image_example(image_radar, image_eth, image_sum_radar, status_radar, status_both, date):
  feature = {
      'image_radar': _array_feature(image_radar),
      'image_eth': _array_feature(image_eth),
      'image_sum_radar': _float_feature(image_sum_radar),
      'exists_radar': _int64_feature(status_radar),
      'exists_both': _int64_feature(status_both),
      'date':_bytes_feature(date)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

def write_day_TRfile(df, file_path):

    if df.shape[0] != 12*24:
        raise AssertionError("Dataframe contains {r} instead of 12*24 rows".format(r=df.shape[0]))
    with tf.io.TFRecordWriter(file_path) as writer:
        for index in range(12*24):
            row = df.iloc[index, :]
            if  bool(row['both_files_okay']):
                image_radar, image_sum_radar = prepare_radar_h5_file(row['file_radar_observed'])
                image_eth = prepare_eth_h5_file(row['file_eth_observed'])
                example_message = image_example(image_radar=image_radar, image_eth=image_eth,
                                                image_sum_radar=image_sum_radar,
                                                status_radar=row['file_radar_okay'], # 1
                                                status_both=row['both_files_okay'], # 1
                                                date=row['date'].encode())

            elif bool(row['file_radar_okay']):
                empty_image = np.full([1, 1], 0)
                image_radar, image_sum_radar = prepare_radar_h5_file(row['file_radar_observed'])
                example_message = image_example(image_radar=image_radar, image_eth=empty_image,
                                                image_sum_radar=image_sum_radar,
                                                status_radar=row['file_radar_okay'], # 1
                                                status_both=row['both_files_okay'], # 0
                                                date=row['date'].encode())

            else:
                empty_image = np.full([1,1],0)
                example_message = image_example(image_radar=empty_image, image_eth=empty_image,
                                                image_sum_radar=0.0,
                                                status_radar=row['file_radar_okay'],  # 0
                                                status_both=row['both_files_okay'],  # 0
                                                date=row['date'].encode())
            writer.write(example_message.SerializeToString())

def create_tfrecord_files(start,end):

    current_date = start
    while current_date <= end:
        print(current_date)
        following_date = current_date + DateOffset(1)
        year, month, day = current_date.strftime('%Y'), current_date.strftime('%m'), current_date.strftime('%d')
        file_names_radar = list(data_dir_radar.glob('{y}/{m}/*{y}{m}{d}*.h5'.format(y=year,m=month, d=day)))
        file_names_eth = list(data_dir_eth.glob('{y}/{m}/*{y}{m}{d}*.h5'.format(y=year,m=month, d=day)))

        files_radar_df = pd.DataFrame({'file_radar': file_names_radar})
        files_radar_df['name_radar'] = files_radar_df['file_radar'].apply(lambda path: path.name)
        files_eth_df = pd.DataFrame({'file_eth': file_names_eth})
        files_eth_df['name_eth'] = files_eth_df['file_eth'].apply(lambda path: path.name)

        expected_dates = [dt.strftime('%Y%m%d%H%M') for dt in pd.date_range(start=current_date, end=following_date, freq="300s")][:-1]
        expected_dates_df = pd.DataFrame({'date': expected_dates})
        expected_dates_df['file_eth'] = expected_dates_df['date'].apply(
            lambda date: data_dir_eth / date[4:6] / ('RAD_NL25_ETH_NA_' + date + '.h5'))
        expected_dates_df['file_radar'] = expected_dates_df['date'].apply(
            lambda date: data_dir_radar / date[4:6] / ('RAD_NL25_RAC_5min_' + date + '_cor.h5'))

        expected_dates_df['name_radar'] = expected_dates_df['file_radar'].apply(lambda path: path.name)
        expected_dates_df['name_eth'] = expected_dates_df['file_eth'].apply(lambda path: path.name)


        merged_df = pd.merge(expected_dates_df, files_radar_df, how='left', left_on='name_radar', right_on='name_radar',
                             suffixes=('_expected', '_observed'))
        merged_df = pd.merge(merged_df, files_eth_df, how='left', left_on='name_eth', right_on='name_eth',
                             suffixes=('_expected', '_observed'))

        print(pd.isna(merged_df['file_radar_observed']).sum(), "radar files not found")
        print(pd.isna(merged_df['file_eth_observed']).sum(), "eth files not found")
        merged_df['file_radar_okay'] = merged_df['file_radar_observed'].apply(check_radar_h5_file)
        merged_df['file_eth_okay'] = merged_df['file_eth_observed'].apply(check_eth_h5_file)
        print(pd.isna(merged_df['file_radar_okay']).sum(), "radar files not valid")
        print(pd.isna(merged_df['file_eth_okay']).sum(), "eth files not valid")
        merged_df['both_files_okay'] =  merged_df['file_radar_okay'] * merged_df['file_eth_okay']
        TFfile_path = out_dir / "{y}/{m}/".format(y=year,m=month)
        TFfile_path.mkdir(parents=True, exist_ok=True)
        shard_name = TFfile_path / "Joined_RAD_NL25_ETH_NA_{y}{m}{d}.tfrecords".format(y=year,m=month,d=day)
        print(merged_df.head())
        write_day_TRfile(merged_df, str(shard_name))
        current_date = following_date

create_tfrecord_files(start,end)

# TODO also calculate average and sd or max for normalizing
