from default import *
import os
chunker = LSTMTagger(os.path.join('data', 'train.txt.gz'), os.path.join('data', 'chunker'), '.tar')
print(chunker.model)