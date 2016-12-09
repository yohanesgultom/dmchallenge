from keras.utils.visualize_util import plot
from keras.applications.vgg16 import VGG16
from keras.models import model_from_json
import sys

# extractor = VGG16(weights='imagenet', include_top=False)
# plot(extractor, to_file='vgg16.png')

arch_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else 'model.png'
with open(arch_file) as f:
    arch_json = f.read()
    model = model_from_json(arch_json)
plot(model, to_file=output_file)
