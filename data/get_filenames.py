from os import listdir
from os.path import isfile, join

path = '/home/ruslan.nikolaev/dev/theia_wrapper/theia/data/imgs/'
filename = '/home/ruslan.nikolaev/dev/theia_wrapper/theia/data/files.txt'

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
filepaths = [join(path, f) for f in sorted(onlyfiles)]

with open(filename, 'w') as file:
    for line in filepaths:
        file.write(line)
        file.write('\n')