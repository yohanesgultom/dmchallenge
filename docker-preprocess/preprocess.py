'''
Build DM dataset
Usage: python preprocess.py <in:DM images directory> <in:DM crosswalk file> <in:DM meta file> <out:meta pickle> <out:dataset h5>

'''
import dicom
import os
import numpy
import sys
import numpy as np
# import pylab
import csv
import scipy.ndimage.interpolation
# import progressbar
import tables
import warnings
import pickle
import math
import multiprocessing
# from sklearn.preprocessing import StandardScaler

# expected width/length (assumed square)
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
EXPECTED_CLASS = 1
MAX_VALUE = 4095

# static
# scaler = StandardScaler()


# preprocess images and append it to a h5 file
def preprocess_images(filedir, filenames, datafilename):
    processname = multiprocessing.current_process().name
    datafile = tables.open_file(datafilename, mode='w')
    data = datafile.create_earray(datafile.root, 'data', tables.Float32Atom(shape=EXPECTED_DIM), (0,), 'dream')
    total = len(filenames)
    count = 0
    for f in filenames:
        data.append(preprocess_image(os.path.join(filedir, f)))
        count += 1
        if count >= 10 and count % 10 == 0:
            print('{}: {}/{}'.format(processname, count, total))
    print('{}: {}/{}'.format(processname, count, total))
    datafile.close()


# preprocess image and return vectorized value
def preprocess_image(filename):
    dcm = dicom.read_file(filename)
    m = center_crop_resize(dcm.pixel_array)
    return np.array([[m, m, m]])


# center crop non-zero and downsample to EXPECTED_SIZE
def center_crop_resize(dat):
    # crop zeros (black parts)
    cropped = crop(dat)
    # center crop:
    # crop by height (y axis) to match width (x axis)
    start = cropped.shape[0] / 2 - (cropped.shape[1] / 2)
    end = start + cropped.shape[1]
    cropped = cropped[start:end, :]
    # resize (downsample)
    scale = EXPECTED_SIZE * 1.0 / cropped.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized = scipy.ndimage.interpolation.zoom(cropped, scale, order=3, prefilter=True)
        assert resized.shape == (EXPECTED_SIZE, EXPECTED_SIZE)
        # scaled to 0..1
        norm = resized * 1.0 / MAX_VALUE
        # # normalize
        # norm = scaler.fit_transform(resized)
    # print(cropped.shape)
    # print(resized.shape)
    # print(resized)
    # print(norm)
    # # render image
    # images = [dat, norm]
    # fig = pylab.figure()
    # for i, m in enumerate(images):
    #     fig.add_subplot(len(images), 1, i + 1)
    #     pylab.imshow(m, cmap=pylab.cm.bone)
    # pylab.show()
    return norm


# Crop non-zero rectangle
# Original http://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
def crop(dat):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(dat)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    # plus 1 because slice isn't inclusive
    return dat[
        top_left[0]:bottom_right[0] + 1,
        top_left[1]:bottom_right[1] + 1
    ]


def parse_int(s):
    try:
        i = int(s)
        return i
    except ValueError:
        return 0


def parse_float(s):
    try:
        i = float(s)
        return i
    except ValueError:
        return 0.0

if __name__ == '__main__':
    dcm_dir = sys.argv[1]
    crosswalk_file = sys.argv[2]
    meta_file = sys.argv[3]
    meta_outfile = sys.argv[4]
    data_outfile = sys.argv[5]
    data_outfile_dir = os.path.dirname(os.path.abspath(data_outfile))

    # pytables file
    datafile = tables.open_file(data_outfile, mode='w')
    data = datafile.create_earray(datafile.root, 'data', tables.Float32Atom(shape=EXPECTED_DIM), (0,), 'dream')
    labels = datafile.create_earray(datafile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'dream')

    # read metadata
    metadata = {}
    with open(meta_file, 'r') as metain:
        reader = csv.reader(metain, delimiter='\t')
        headers = next(reader, None)
        for row in reader:
            key = row[0] + '_' + row[1]
            metadata[key] = {
                'id': row[0],
                'examIndex': row[1],
                'daysSincePreviousExam': parse_int(row[2]),
                'cancerL': parse_int(row[3]),
                'cancerR': parse_int(row[4]),
                'invL': parse_int(row[5]),
                'invR': parse_int(row[6]),
                'age': parse_int(row[7]),
                'implantEver': parse_int(row[8]),
                'implantNow': parse_int(row[9]),
                'bcHistory': parse_int(row[10]),
                'yearsSincePreviousBc': parse_float(row[11]),
                'previousBcLaterality': parse_int(row[12]),
                'reduxHistory': parse_int(row[13]),
                'reduxLaterality': parse_int(row[14]),
                'hrt': parse_int(row[15]),
                'antiestrogen': parse_int(row[16]),
                'firstDegreeWithBc': parse_int(row[17]),
                'firstDegreeWithBc50': parse_int(row[18]),
                'bmi': parse_float(row[19]),
                'race': parse_int(row[20])
            }

    # read crosswalk
    filenames = []
    with open(crosswalk_file, 'rb') as tsvin:
        crosswalk = csv.reader(tsvin, delimiter='\t')
        headers = next(crosswalk, None)
        for row in crosswalk:
            dcm_subject_id = row[0]
            dcm_exam_id = row[1]
            dcm_laterality = row[4]
            dcm_filename = row[5]
            key = row[0] + '_' + row[1]
            dcm_label = metadata[key]['cancer' + dcm_laterality.upper()]
            filenames.append(dcm_filename)
            labels.append(np.array([[dcm_label]]))
            # print('meta[{}][{}] = {}'.format(key, 'cancer' + dcm_laterality.upper(), dcm_label))

    # read dicom images parallelly
    # use half of available "cpus"
    cpu_count = multiprocessing.cpu_count()
    chunk_size = int(math.ceil(len(filenames) * 1.0 / cpu_count))
    tmp_names = []
    processes = []
    for i in range(cpu_count):
        tmp_names.append(os.path.join(data_outfile_dir, 'tmp{}.h5'.format(i)))
        start = i * chunk_size
        end = start + chunk_size
        p = multiprocessing.Process(name=tmp_names[i], target=preprocess_images, args=(dcm_dir, filenames[start:end], tmp_names[i]))
        p.start()
        processes.append(p)
    # wait all processes to complete
    for p in processes:
        p.join()
    # merge tmp files arrays to single array
    # and delete all tmp files
    for f in tmp_names:
        datafile = tables.open_file(f, mode='r')
        data.append(datafile.root.data[:])
        datafile.close()
        os.remove(f)

    print(data[:].shape)
    print(labels[:].shape)
    assert data[:].shape == (len(filenames), EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
    assert labels[:].shape == (len(filenames), EXPECTED_CLASS)

    # save metadata
    with open(meta_outfile, 'wb') as handle:
        pickle.dump(metadata, handle)

    # close file
    datafile.close()
