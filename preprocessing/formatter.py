import os
from pathlib import Path
import re

import spacy

from parameters import *
from regular_exp import remove_multiple_new_lines_and_spaces


def format_file(filepath, model):
    file_text = open(filepath, 'r', encoding='utf-8').read()

    segmented_file = list(model(file_text).sents)
    formatted_file = '\n'.join([str(x).replace('\n', ' ') for x in segmented_file])

    formatted_file = re.split(r'(\[\d{1,4}\])', formatted_file)
    formatted_file = '\n'.join(formatted_file)
    formatted_file = remove_multiple_new_lines_and_spaces(formatted_file)
    return formatted_file


def format_directory(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    for idx, filename in enumerate(os.listdir(input_folder)):
        formatted_file = format_file(Path.joinpath(Path(input_folder), Path(filename)), model)
        out_path = Path.joinpath(Path(output_folder), Path(filename))

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(formatted_file)

        if (idx + 1) % 100 == 0:
            print(f'Processed {idx+1} files')


if __name__ == '__main__':
    folder = 'Dataset/regex_preprocessed_%s' % PREPROCESSING_DATASET_TYPE
    output_folder = 'Dataset/sentence_preprocessing_%s' % PREPROCESSING_DATASET_TYPE
    os.makedirs(output_folder, exist_ok=True)

    model = spacy.load('xx_sent_ud_sm')
    format_directory(folder, output_folder, model)

    # filename = '003523.txt'
    # result = format_file(os.path.join(folder, filename))
