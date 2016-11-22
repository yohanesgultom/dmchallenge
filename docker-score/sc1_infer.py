import sys
import os
import csv

PREDICTIONS_PATH = 'predictions.tsv'

dcm_dir = sys.argv[1]
crosswalk_file = sys.argv[2]
meta_file = sys.argv[3]
weights_file = sys.argv[4]
predictions_file = sys.argv[5] if len(sys.argv) > 5 else PREDICTIONS_PATH

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


predictions = [('1', 'L', '0.99'), ('1', 'L', '0.99'), ('1', 'L', '0.99'), ('1', 'R', '0.88')]

# write predictions
with open(predictions_file, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t')
    spamwriter.writerow(['subjectId', 'laterality', 'confidence'])
    for p in predictions:
        spamwriter.writerow(p)
