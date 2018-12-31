import xml.etree.ElementTree as ET
import os

from shutil import copyfile

DETRAC_ann_base = '/home/nikolaevra/datasets/traffic/DETRAC-Train-Annotations-XML'
DETRAC_data_base = '/home/nikolaevra/datasets/traffic/Insight-MVT_Annotation_Train'
output_dir_base = '/home/nikolaevra/datasets/traffic/parsed_data'

IMG_WIDTH = 960.0
IMG_HEIGHT = 540.0

LIMIT = 50


def convert_to_yolo_box(height, width, left, top):
    x_center = left + (width / 2.)
    y_center = top + (height / 2.)

    return [
        x_center / IMG_WIDTH,
        y_center / IMG_HEIGHT,
        width / IMG_WIDTH,
        height / IMG_HEIGHT
    ]


def get_all_filenames(folder_name):
    return_list = []

    for image_file in os.listdir(os.path.join(DETRAC_data_base, folder_name)):
        if image_file.endswith(".jpg"):
            return_list.append(image_file)

    return return_list


def save_seqs_and_copy_images(sequences, imgs_folder, all_img_filenames):
    print("Writing {} files.".format(len(sequences)))

    if len(sequences) > 0:
        for filename, seq in sequences.iteritems():
            with open(os.path.join(output_dir_base, filename[:-3] + 'txt'), 'w') as f:
                f.write("\n".join(z for z in [" ".join([str(y) for y in x]) for x in seq]))

        for old_filename in all_img_filenames:
            copyfile(
                os.path.join(os.path.join(DETRAC_data_base, imgs_folder), old_filename),
                os.path.join(output_dir_base, imgs_folder + '_' + old_filename)
            )


def parse_single_file(dir_name, categories, all_img_filenames):
    parsed_xml = ET.parse(os.path.join(DETRAC_ann_base, dir_name) + '.xml').getroot()
    to_write_name_labels_seq = {dir_name + '_' + k: [] for k in all_img_filenames}

    for img_filename in all_img_filenames:
        img_id = int(img_filename[3:8])
        found_targets = [x for x in parsed_xml.iter('frame') if x.attrib['num'] == str(img_id)]

        if len(found_targets) > 0:
            for target in found_targets[0].iter('target'):
                vehicle_type = target.find('attribute').get('vehicle_type')

                if vehicle_type not in categories:
                    obj_id = len(categories)
                    categories[vehicle_type] = obj_id

                to_write_name_labels_seq[dir_name + '_' + img_filename].append(
                    [categories[vehicle_type]] +
                    convert_to_yolo_box(
                        float(target.find('box').get('height')),
                        float(target.find('box').get('width')),
                        float(target.find('box').get('left')),
                        float(target.find('box').get('top'))
                    )
                )
        else:
            print("Didn't find a matching target.")

    return to_write_name_labels_seq


def find_all_folder_names(dir_name):
    return [os.path.basename(x[0]) for x in os.walk(dir_name)]


if __name__ == '__main__':
    cat_map = {}
    image_filenames = []

    folders = find_all_folder_names(DETRAC_data_base)[1:]

    for folder in folders:
        print("Processing {} folder.".format(folder))

        all_filenames = get_all_filenames(folder)
        all_filenames = all_filenames[:LIMIT]

        seqs = parse_single_file(folder, cat_map, all_filenames)
        save_seqs_and_copy_images(seqs, folder, all_filenames)

        image_filenames += seqs.keys()

    with open(os.path.join(output_dir_base, 'image_names.txt'), 'w') as f:
        f.write('\n'.join(image_filenames))

    with open(os.path.join(output_dir_base, 'obj.names'), 'w') as f:
        f.write('\n'.join(cat_map.keys()))
