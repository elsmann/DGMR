

import math
from datetime import timedelta,datetime
import h5py
import numpy as np
import os
from collections import deque
import random
import tensorflow as tf
random.seed(10)

def daterange(start_date, end_date, step):
    current = start_date
    while current < end_date:
        yield current
        current += step

def amount_of_days_in_month(year, month):
    d = datetime(year + int(month / 12), month % 12 + 1, 1) - timedelta(days=1)
    return int(d.strftime('%d'))

def crop_size(file_directory):
    """
    calculates index of rows and columns for 256x256 crops
    :return: tuple of int
        (upper_row,lowest_row,left_column,right_column)
         """
    direc= os.path.join(file_directory, "2008", "02")
    f = os.path.join(direc, sorted(os.listdir(direc))[0])
    file = h5py.File(f, 'r')
    image =  np.asarray(file['image1']['image_data'])
    rows_with_values = np.argwhere(image <65535)
    upper_row = rows_with_values[0,0]
    lowest_row = rows_with_values[-1,0]
    image= image.transpose()
    rows_with_values = np.argwhere(image <65535)
    left_column = rows_with_values[0,0]
    right_column = rows_with_values[-1,0]
    length_rows = lowest_row-upper_row
    upper_nrow = int(upper_row + ((length_rows - 256) / 2))
    lower_nrow = int(lowest_row - ((length_rows - 256) / 2))
    length_columns = right_column-left_column
    left_ncolumn = int(left_column + ((length_columns - 256) / 2))
    right_ncolumn = int(right_column - ((length_columns - 256) / 2))
    print("Created crop of size: ", lower_nrow-upper_nrow, right_ncolumn-left_ncolumn)
    file.close()
    return(upper_nrow, lower_nrow, left_ncolumn, right_ncolumn)

def prepare_rad_h5_file(file, upper_row=300, lowest_row=556, left_column=241, right_column=497 ):
    f = h5py.File(file, 'r')
    try:
        image = f['image1']['image_data']
        image = np.asarray(image, dtype=np.float32)
        f.close()
        image = image[upper_row:lowest_row, left_column:right_column]
        image = np.where( image == 65535, np.NaN, image / 100.0 )

    except:
        print("couldn't open image data of", file)
        image = np.full([lowest_row-upper_row, right_column-left_column], np.nan)

    return image


def create_dataset(file_directory, debugging=False):
    if not debugging:
        years = ["2008", "2009", "2010", "2011", "2012", "2013"]
        months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
        # skip first day of month
        first_day = 2
    else:
        years = ["2008"]
        months = ["01"]
        first_day = 29
    upper_row, lowest_row, left_column, right_column = crop_size(file_directory)
    datapoints =np.empty((717700,22,256,256), dtype=np.float32)
    index = 0
    for year in years:
        directory_year = os.path.join(file_directory, year)
        for element in months:
            print("Month ", int(element))
            directory_month = os.path.join(directory_year,element)
            dates = sorted([dt.strftime('%Y%m%d%H%M') for dt in
                daterange(datetime(int(year), int(element), first_day,0,0), datetime(int(year), int(element),  amount_of_days_in_month(int(year),int(element)),23,59),timedelta(minutes=5))])
            sequence = deque()
            moving_sum = 0
            for filename in dates:
                rad_file = 'RAD_NL25_RAC_5min_' + str(filename) + '_cor.h5'
                rad_file = os.path.join(directory_month,  rad_file)
                image = prepare_rad_h5_file(rad_file,upper_row,lowest_row,left_column,right_column)
                if len(sequence) < 22:
                    sequence.append(image)
                    moving_sum += np.sum(image)
                if len(sequence) == 22:
                    moving_sum += np.sum(image) - np.sum(sequence.popleft())
                    sequence.append(image)
                    if np.isnan(np.sum(image)):
                        moving_sum = 0
                        sequence.clear()
                    else:
                        print(moving_sum)
                        prob = 1 - math.exp(-(moving_sum/500))
                        prob = min(1, prob+ 0.002)
                        if prob > random.random():
                            np_sequence = np.asanyarray(sequence)
                            datapoints[index] = np_sequence
                            index += 1

    datapoints = datapoints[:index]
    print("shape of dataset:", datapoints.shape)
    dataset = tf.data.Dataset.from_tensor_slices(datapoints)
    return dataset

#tf_dataset = create_dataset('/Users/frederikesmac/important stuff/Uni/MA/Data/data/RAD_NL25_RAC_5min/', True)
#tf_data_path = "/Users/frederikesmac/important stuff/Uni/MA/Data/data/dataset_31Jan2018"
#tf.data.experimental.save( tf_dataset, tf_data_path, compression='GZIP')

