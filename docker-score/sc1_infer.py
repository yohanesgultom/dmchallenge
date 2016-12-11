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
from preprocess import preprocess_image, EXPECTED_DIM, MAX_VALUE, FILTER_THRESHOLD

PREDICTIONS_PATH = 'predictions.tsv'

dcm_dir = sys.argv[1]
scratch_dir = sys.argv[2]
crosswalk_file = sys.argv[3]
arch_file = sys.argv[4]
weights_file = sys.argv[5]
predictions_file = sys.argv[6] if len(sys.argv) > 6 else PREDICTIONS_PATH

print('Loading extractor and model')
# feature extractor
extractor = VGG16(weights='imagenet', include_top=False)
# load model
with open(arch_file) as f:
    arch_json = f.read()
    model = model_from_json(arch_json)
model.load_weights(weights_file)

# predict images in crosswalk
print('Predicting image by image')
predictions = {}
prediction_index = []
with open(crosswalk_file, 'rb') as tsvin:
    crosswalk = csv.reader(tsvin, delimiter='\t')
    headers = next(crosswalk, None)
    count = 1
    for row in crosswalk:
        # no exam id col for testing
        dcm_subject_id = row[0]
        dcm_laterality = row[3]
        dcm_filename = row[4]
        data = preprocess_image(os.path.join(dcm_dir, dcm_filename), dcm_laterality)
        features = extractor.predict(data)
        prediction = model.predict(features)
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
        # confidence = max(pred['p'])  # max
        confidence = pred['p'][0]  # first
        # confidence = pred['p'][-1]  # latest
        row = (pred['id'], pred['lat'], confidence)
        spamwriter.writerow(row)

print('Done.')
