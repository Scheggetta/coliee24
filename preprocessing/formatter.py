import os
from pathlib import Path
import spacy
import translator as trs


def format_file(dataset_folder, filename):
    file_text = open(Path.joinpath(Path(dataset_folder), Path(filename)), 'r', encoding='utf-8').read()
    # nlp = spacy.load('en_core_web_sm')
    # fr_nlp = spacy.load('fr_core_news_sm')
    multi_nlp = spacy.load('xx_sent_ud_sm')
    segmented_file = list(multi_nlp(file_text).sents)
    # return '\n'.join([str(x).replace('\n', ' ') for x in segmented_file])
    return [str(x).replace('\n', ' ') for x in segmented_file]


def format_directory(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        formatted_file = format_file(input_folder, filename)
        out_path = Path.joinpath(Path(output_folder), Path(filename))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(formatted_file)


if __name__ == '__main__':
    dataset_path = 'Dataset/regex_preprocessed_train_bis'
    filename = '003523.txt'
    formatted_file = format_file(dataset_path, filename)
    with open('segmented_file.txt', 'w', encoding='utf-8') as f:
        f.write(formatted_file)

    trs.compare_french_english_script('segmented_file.txt', 'translated_file.txt')
