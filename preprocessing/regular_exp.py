from pathlib import Path
import os
import re
import time

# TODO choose between Natasha and Spacy for sentence segmentation


def remove_multiple_alien_topics(text):  # TODO: to be tested
    aliens_topic_splits = re.split(r'aliens - topic (\d+(\.{1,}\d+){0,})', text, flags=re.IGNORECASE)
    last_topic = aliens_topic_splits[-1]
    last_topic_splits = re.split(r'\[\d{1,4}\]', last_topic, flags=re.IGNORECASE)
    return '\n'.join([aliens_topic_splits[0]] + last_topic_splits[1:])


def replace_tags_from_regex(text):
    regex_to_be_replaced = [r'\[([A-Z]+?)\]',
                            r'\[(18\d{2})\]',
                            r'\[(19\d{2})\]',
                            r'\[(20\d{2})\]'
                           ]
    regex_to_text = '|'.join(regex_to_be_replaced)
    return re.sub(regex_to_text, '/1', text)


def remove_tags_from_regex(text):
    regex_to_be_removed = [r'\[(translation|traduction)\]',
                           r'\[emphasis added\]',
                           r'\[redacted\]',
                           r'\[see footnote \d{1,}\]'
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


def get_bracket_freqs_dataset(directory='../Dataset/task1_train_files_2024'):
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
    return freq_dict


def remove_least_frequent_square_brackets(text):
    freqs = get_bracket_freqs_dataset()
    regex_to_be_removed = '|'.join([key for key in freqs.keys() if freqs[key] < 50]).replace('[', '\[').replace(']', '\]').replace('(', '\(').replace(')', '\)')
    return re.sub(regex_to_be_removed, '', text)


def remove_multiple_new_lines(text):
    processed_file = re.sub(r'(\n{2,})|(\s\n){2,}', '\n', text)
    processed_file = re.sub(r'(\n\s*)', '\n', processed_file)
    processed_file = re.sub(r'(^\n{1,})', '', processed_file)
    return processed_file


# understand if it is required
def sub_multiple_suppressed_pattern(text):  # TODO: to be tested
    processed_text = re.sub(r'.{0,1}\w*_suppressed.{0,1}', ' OMISSIS ', text, flags=re.IGNORECASE)
    return re.sub(r'(.{0,1}OMISSIS.{0,1}){2,}', ' OMISSIS ', processed_text)


def regex_preprocessing_single_file(filepath):
    file_text = open(filepath).read()
    preprocessed_file = remove_multiple_alien_topics(file_text)
    preprocessed_file = remove_least_frequent_square_brackets(preprocessed_file)
    preprocessed_file = remove_tags_from_regex(preprocessed_file)
    preprocessed_file = replace_tags_from_regex(preprocessed_file)
    return remove_multiple_new_lines(preprocessed_file)


def regex_preprocessing(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    start = time.time()

    for idx, file_name in enumerate(os.listdir(input_directory)):
        file_text = open(Path.joinpath(Path(input_directory), Path(file_name))).read()

        preprocessed_file = remove_multiple_alien_topics(file_text)
        preprocessed_file = remove_least_frequent_square_brackets(preprocessed_file)
        preprocessed_file = remove_tags_from_regex(preprocessed_file)
        preprocessed_file = replace_tags_from_regex(preprocessed_file)
        preprocessed_file = remove_multiple_new_lines(preprocessed_file)

        with open(Path.joinpath(Path(output_directory), Path(file_name)), 'w') as file:
            file.write(preprocessed_file)

        if (idx+1) % 100 == 0:
            print(f'Processed {idx+1} files')
            print(f'Elapsed time: {time.time() - start}')


for filename in os.listdir('../Dataset/regex_preprocessed_train')[:10]:
    path = Path.joinpath(Path('../Dataset/regex_preprocessed_train'), Path(filename))
    file = open(path, 'r', encoding='utf-8')
    text = file.read()
    preprocessed_text = re.sub(r'\((.|\n)*?\)', '', text, flags=re.IGNORECASE)
    file.close()
    with open(path, 'w', encoding='utf-8') as file:
        file.write(preprocessed_text)
