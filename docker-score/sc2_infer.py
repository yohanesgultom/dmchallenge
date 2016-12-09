'''
Use trained model to predict image in crosswalk file
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sc1_infer.py <in:dcm dir> <out:temp output dir> <in:crosswalk file> <in:model architecture json file> <in:model weights h5 file> <out:prediction result tsv>

'''
import sys
import os
import csv
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from preprocess import preprocess_image, metadata2numpy, normalize_meta, parse_int, parse_float, EXPECTED_DIM, MAX_VALUE, FILTER_THRESHOLD

PREDICTIONS_PATH = 'predictions.tsv'

dcm_dir = sys.argv[1]
scratch_dir = sys.argv[2]
crosswalk_file = sys.argv[3]
meta_file = sys.argv[4]
arch_file = sys.argv[5]
weights_file = sys.argv[6]
predictions_file = sys.argv[7] if len(sys.argv) > 7 else PREDICTIONS_PATH

print('Loading extractor and model')
# feature extractor
extractor = VGG16(weights='imagenet', include_top=False)
# load model
with open(arch_file) as f:
    arch_json = f.read()
    model = model_from_json(arch_json)
model.load_weights(weights_file)

print('Predicting images one by one')
metadata = {}
predictions = {}
prediction_index = []

# read exam metadata
with open(meta_file, 'r') as metain:
    reader = csv.reader(metain, delimiter='\t')
    headers = next(reader, None)
    for row in reader:
        meta_key = '{}_{}'.format(row[0], row[1])
        metadata[meta_key] = {
            'id': row[0],
            'examIndex': row[1],
            'daysSincePreviousExam': normalize_meta(row, 2, 'daysSincePreviousExam'),
            'age': normalize_meta(row, 3, 'age'),
            'implantEver': normalize_meta(row, 4, 'implantEver'),
            'implantNow': normalize_meta(row, 5, 'implantNow'),
            'bcHistory': normalize_meta(row, 6, 'bcHistory'),
            'yearsSincePreviousBc': normalize_meta(row, 7, 'yearsSincePreviousBc'),
            'previousBcLaterality': normalize_meta(row, 8, 'previousBcLaterality'),
            'reduxHistory': normalize_meta(row, 9, 'reduxHistory'),
            'reduxLaterality': normalize_meta(row, 10, 'reduxLaterality'),
            'hrt': normalize_meta(row, 11, 'hrt'),
            'antiestrogen': normalize_meta(row, 12, 'antiestrogen'),
            'firstDegreeWithBc': normalize_meta(row, 13, 'firstDegreeWithBc'),
            'firstDegreeWithBc50': normalize_meta(row, 14, 'firstDegreeWithBc50'),
            'bmi': normalize_meta(row, 15, 'bmi'),
            'race': normalize_meta(row, 16, 'race')
        }

# predict images in crosswalk
with open(crosswalk_file, 'rb') as tsvin:
    crosswalk = csv.reader(tsvin, delimiter='\t')
    headers = next(crosswalk, None)
    count = 1
    for row in crosswalk:
        dcm_subject_id = row[0]
        dcm_exam_index = row[1]
        dcm_laterality = row[4]
        dcm_filename = row[5]
        # extract features from image
        data = preprocess_image(os.path.join(dcm_dir, dcm_filename), dcm_laterality)
        features = extractor.predict(data)
        # get metadata features
        meta_key = '{}_{}'.format(dcm_subject_id, dcm_exam_index)
        meta = metadata2numpy(metadata[meta_key])
        prediction = model.predict([features, meta])
        # collect predictions based on subjectId and laterality
        # to handle duplications
        key = '{}_{}'.format(dcm_subject_id, dcm_laterality)
        if key not in predictions:
            predictions[key] = {
                'id': dcm_subject_id,
                'lat': dcm_laterality,
                'p': []
            }
            # maintain order by storing index
            prediction_index.append(key)
        predictions[key]['p'].append(prediction[0][0])
        count += 1

# write predictions
print('Writing to result {}'.format(predictions_file))
with open(predictions_file, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t')
    header = ('subjectId', 'laterality', 'confidence')
    spamwriter.writerow(['subjectId', 'laterality', 'confidence'])
    # iterate index to keep order
    for key in prediction_index:
        pred = predictions[key]
        # aggregation strategy
        # confidence = sum(pred['p']) / float(len(pred['p']))  # average
        confidence = max(pred['p'])  # max
        # confidence = pred['p'][-1]  # latest
        row = (pred['id'], pred['lat'], confidence)
        spamwriter.writerow(row)

print('Done.')
