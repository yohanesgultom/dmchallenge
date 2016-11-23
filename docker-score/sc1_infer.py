import sys
import os
import csv
from keras.models import model_from_json
from preprocess import preprocess_image, EXPECTED_DIM, MAX_VALUE, FILTER_THRESHOLD

PREDICTIONS_PATH = 'predictions.tsv'

dcm_dir = sys.argv[1]
scratch_dir = sys.argv[2]
crosswalk_file = sys.argv[3]
arch_file = sys.argv[4]
weights_file = sys.argv[5]
predictions_file = sys.argv[6] if len(sys.argv) > 6 else PREDICTIONS_PATH

# load model
print('Loading model')
with open(arch_file) as f:
    arch_json = f.read()
    model = model_from_json(arch_json)
model.load_weights(weights_file)

# predict images in crosswalk
print('Predicting image by image')
predictions = []
with open(crosswalk_file, 'rb') as tsvin:
    crosswalk = csv.reader(tsvin, delimiter='\t')
    headers = next(crosswalk, None)
    count = 1
    for row in crosswalk:
        dcm_subject_id = row[0]
        dcm_exam_id = row[1]
        dcm_laterality = row[4]
        dcm_filename = row[5]
        data = preprocess_image(os.path.join(dcm_dir, dcm_filename), EXPECTED_DIM[1], MAX_VALUE, FILTER_THRESHOLD)
        prediction = model.predict(data)
        p = (dcm_subject_id, dcm_laterality, prediction[0][0])
        print('{} {}'.format(count, p))
        predictions.append(p)
        count += 1

# write predictions
print('Writing to result {}'.format(predictions_file))
with open(predictions_file, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t')
    header = ('subjectId', 'laterality', 'confidence')
    spamwriter.writerow(['subjectId', 'laterality', 'confidence'])
    for p in predictions:
        spamwriter.writerow(p)

print('Done.')
