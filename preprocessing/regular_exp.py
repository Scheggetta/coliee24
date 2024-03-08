from pathlib import Path
import os
import re
import time

from parameters import *


# TODO:
#  - spacy or natasha
#  - check translations
#  - check gpt embeddings


def remove_file_header(text):
    processed_file = re.split(r'\[1\n?\]', text)
    return '[1]\n' + ''.join(processed_file[1:])


def remove_multiple_alien_topics(text):
    aliens_topic_splits = re.split(r'.* - topic (\d+(\.{1,}\d+){0,})', text, flags=re.IGNORECASE)

    if len(aliens_topic_splits) == 1:
        return text
    aliens_topic_splits = [el for el in aliens_topic_splits if el is not None]

    last_topic = aliens_topic_splits[-1]
    last_topic_splits = re.split(r'(\[\d{1,4}\])', last_topic, flags=re.IGNORECASE)
    return '\n'.join([aliens_topic_splits[0]] + last_topic_splits[1:])


def remove_least_frequent_square_brackets(text, regex_to_be_removed):
    return re.sub(regex_to_be_removed, '', text)


def remove_tags_from_regex(text):
    regex_to_be_removed = [r'\((.|\n)*?\)',
                           r'\[(translation|traduction)\]',
                           r'\[emphasis added\]',
                           r'\[redacted\]',
                           r"\[Non souligné dans l'original\]",
                           r'\[the applicant\]',
                           r'\[the act\]',
                           r'\[citation omitted\]',
                           r'\[the board\]',
                           r'\[English language version follows French language version\]',
                           r'\[French language version follows English language version\]',
                           r'\[La version française vient à la suite de la version anglaise\]',
                           r'\[La version anglaise vient à la suite de la version française\]',
                           r'MLB headnote and full text',
                           r'MLB unedited judgment',
                           r'\[see footnote \d{1,}\]',
                           r'\[para\. \d{1,}.*?\]',
                           r'\[sic\]',
                           r'\[End of document\]',
                           r'(Editor:.*\n)|(Editor:.*$)',
                           r'(\[\s{0,}\.\s{0,}\.\s{0,}\.\s{0,}\])',
                           r'((\.\s{0,}){4,})',
                           r'[^\u0000-\u007E\u00A1-\u00AC\u00AE-\u01FF]',
                           r'[\[<(]?[ \t]*\w*_suppressed[ \t]*[\]>)]?'
                          ]
    regex_to_text = '|'.join(regex_to_be_removed)
    return re.sub(regex_to_text, '', text, flags=re.IGNORECASE)


def replace_tags_from_regex(text):
    preprocessed_file = re.sub(r'\[([A-Z]+?)\]', r'\1', text)
    preprocessed_file = re.sub(r'\[([a-z])\]', r'\1', preprocessed_file)
    preprocessed_file = re.sub(r'\[(18\d{2})\]', r'\1', preprocessed_file)
    preprocessed_file = re.sub(r'\[(19\d{2})\]', r'\1', preprocessed_file)
    preprocessed_file = re.sub(r'\[(20\d{2})\]', r'\1', preprocessed_file)

    return preprocessed_file


def remove_multiple_new_lines_and_spaces(text):
    processed_file = re.sub(r'(\n{2,})|(\s\n){2,}', '\n', text)
    processed_file = re.sub(r'(\n\s*)', '\n', processed_file)
    processed_file = re.sub(r'(^\n{1,})', '', processed_file)
    processed_file = re.sub(r'( {2,})', ' ', processed_file)
    processed_file = re.sub(r' *\n', '\n', processed_file)
    return processed_file


def get_unfrequent_square_brackets_regex(directory):
    freq_dict = dict()
    for file in os.listdir(directory):
        with open(Path.joinpath(Path(directory), Path(file)), 'r', encoding='utf-8') as f:
            text = f.read()
            list_found_keys = re.findall(r'\[.*?\]', text)
            for key in list_found_keys:
                if key in freq_dict.keys():
                    freq_dict[key] += 1
                else:
                    freq_dict[key] = 1

    regex_to_be_removed = []
    for key in freq_dict.keys():
        if freq_dict[key] < 50 and not bool(re.fullmatch(r'\[\d{1,4}\]', key)):
            regex_to_be_removed.append(re.escape(key))
    regex_to_be_removed = '|'.join(regex_to_be_removed)

    return regex_to_be_removed


def regex_preprocessing_single_file(folder, filename):
    filepath = Path.joinpath(Path(folder), Path(filename))
    usb_regex = get_unfrequent_square_brackets_regex(folder)

    file_text = open(filepath).read()
    preprocessed_file = remove_file_header(file_text)
    preprocessed_file = replace_tags_from_regex(preprocessed_file)
    preprocessed_file = remove_least_frequent_square_brackets(preprocessed_file, usb_regex)
    preprocessed_file = remove_tags_from_regex(preprocessed_file)
    return remove_multiple_new_lines_and_spaces(preprocessed_file)


def regex_preprocessing(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    usb_regex = get_unfrequent_square_brackets_regex(input_directory)

    start = time.time()

    for idx, file_name in enumerate(os.listdir(input_directory)):
        file_text = open(Path.joinpath(Path(input_directory), Path(file_name))).read()

        preprocessed_file = remove_file_header(file_text)
        preprocessed_file = replace_tags_from_regex(preprocessed_file)
        preprocessed_file = remove_least_frequent_square_brackets(preprocessed_file, usb_regex)
        preprocessed_file = remove_tags_from_regex(preprocessed_file)
        preprocessed_file = remove_multiple_new_lines_and_spaces(preprocessed_file)

        with open(Path.joinpath(Path(output_directory), Path(file_name)), 'w') as file:
            file.write(preprocessed_file)

        if (idx + 1) % 100 == 0:
            print(f'Processed {idx+1} files')
            print(f'Elapsed time: {time.time() - start}')


if __name__ == '__main__':
    folder = 'Dataset/task1_%s_files_2024' % PREPROCESSING_DATASET_TYPE

    # result = find_paragraph1_occurrences(folder)
    # for filename in os.listdir(folder)[:10]:

    filename = '089775.txt'
    result = regex_preprocessing_single_file(folder, filename)
    # Write the result to a file
    with open('preprocessed.txt', 'w') as f:
        f.write(result)
