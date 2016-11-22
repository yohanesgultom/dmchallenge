import sys
import os
import csv
from keras.models import model_from_json

PREDICTIONS_PATH = 'predictions.tsv'

dcm_dir = sys.argv[1]
scratch_dir = sys.argv[2]
crosswalk_file = sys.argv[3]
meta_file = sys.argv[4]
arch_file = sys.argv[5]
weights_file = sys.argv[6]
predictions_file = sys.argv[7] if len(sys.argv) > 7 else PREDICTIONS_PATH

# only for subchallenge 2
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

# load model
model = model_from_json(arch_file)
model.load_weights(weights_file)

# predictions = [('1', 'L', '0.99'), ('1', 'L', '0.99'), ('1', 'L', '0.99'), ('1', 'R', '0.88')]
predictions = []
# TODO

# write predictions
with open(predictions_file, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t')
    spamwriter.writerow(['subjectId', 'laterality', 'confidence'])
    for p in predictions:
        spamwriter.writerow(p)
