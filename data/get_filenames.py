from os import listdir
from os.path import isfile, join

path = '/home/nikolaevra/dev/detectors/data/train_data/custom_data'
filename = '/home/nikolaevra/dev/detectors/data/files.txt'

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
filepaths = [join(path, f) for f in sorted(onlyfiles)]

with open(filename, 'w') as file:
    for line in filepaths:
        if line[-4:] == '.jpg':
            file.write(line)
            file.write('\n')