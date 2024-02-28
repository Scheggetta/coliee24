from pathlib import Path
import os
import re


# TODO:
#  - get the year of the document
#  - assess what's inside the () tags
#  - preprocessing to remove useless new lines, <FRAGMENT_SUPPRESSES>, etc... -> Generalize it in the case of [ something ] as well
#  - there are other non visible characters that should be removed, like  
#  - remove frequent notes, like `[End of document]`, `[Translation]`, `MLB headnote and full text`, `This case is unedited, therefore contains no summary.`, `[French language version follows English language version]`, `[La version française vient à la suite de la version anglaise]`, `MLB unedited judgment`, etc...
#  - consider the subsection (1), (2), etc.. problem. Example: 069639.txt

# TODO:
#   - '[...]' (all variations) ----> DONE
#   - Translation (all variations fr as well)
#   - [[a-z]] are typos -> substitute with [a-z] only (generalize to more than one character example [the]) ----> DONE
#   - [see footnote [0-9]] ----> DONE
#   - [emphasis added] (all variations) ---> DONE
#   - [sic] ---> DONE
#   - adjust something_suppressed detection
#   - [redacted] ----> DONE
#   - [acronyms] -> acronyms such as ([IRPA] -> IRPA, [RPD] -> RPD) ----> DONE
#   - [para. [0-9]*] ----> DONE
#   - [and also with] -> and also with
#   - [[a-zA-Z]] to be done at the end of the preprocessing being sure that everything else is OK(to be revised)
#   - [Non souligné dans l'original]
#   - [the applicant] (all variations) -> the applicant
#   - [English language version follows French language version] (all variations + fr)
#   - [the act] -> the act (all variations)
#   - [citation omitted]
#   - [the board] (all variations) -> the board
#   - remove everything that has a frequency less than 50 in the square brackets ----> DONE

def remove_multiple_alien_topics(text):  # TODO: to be tested
    aliens_topic_splits = re.split(r'aliens - topic (\d+(\.{1,}\d+){0,})', text, flags=re.IGNORECASE)
    last_topic = aliens_topic_splits[-1]
    last_topic_splits = re.split(r'\[\d{1,4}\]', last_topic, flags=re.IGNORECASE)
    return '\n'.join([aliens_topic_splits[0]] + last_topic_splits[1:])


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


def remove_translation_tag(text):
    return re.sub(r'\[(translation|traduction)\]', '', text, flags=re.IGNORECASE)


def remove_emphasis_added_tag(text):
    return re.sub(r'\[emphasis added\]', '', text, flags=re.IGNORECASE)


def replace_acronyms(text):
    return re.sub(r'\[([A-Z]+?)\]', r'\1', text)


def remove_redacted_tag(text):
    return re.sub(r'\[redacted\]', '', text, flags=re.IGNORECASE)


def remove_references(text):
    processed_text = re.sub(r'\[see footnote \d{1,}\]', '', text, flags=re.IGNORECASE)
    return re.sub(r'\[para\. \d{1,}.*?\]', '', processed_text)


def remove_reported_typos(text):
    return re.sub(r'\[sic\]', '', text, flags=re.IGNORECASE)


def correct_known_typos(text):
    return re.sub(r'\[([a-zA-Z])\]', r'\1', text)


def remove_least_frequent_square_brackets(text):
    freqs = get_bracket_freqs_dataset()
    for key in freqs.keys():
        if freqs[key] < 50:
            text = text.replace(key, '')
    return text


def remove_multiple_spaces(text):
    return re.sub(r'( {2,}|\t{1,})', ' ', text)


def remove_multiple_new_lines(text):
    processed_file = re.sub(r'(\n{2,})|(\s\n){2,}', '\n', text)
    processed_file = re.sub(r'(\n\s)', '\n', processed_file)
    processed_file = re.sub(r'(^\n{1,})', '', processed_file)
    return processed_file


def remove_year_brackets(text):
    text = re.sub(r'\[(18\d{2})\]', r'\1', text)
    text = re.sub(r'\[(19\d{2})\]', r'\1', text)
    text = re.sub(r'\[(20\d{2})\]', r'\1', text)
    return text


def remove_end_file(text):
    return re.sub(r'\[End of document\]', '', text)


def remove_editor_name(text):
    return re.sub(r'(Editor:.*\n)|(Editor:.*$)', '', text)


def remove_suppressed_pattern(text): # TODO: needs to be executed before the spaces and tabs removal
    return re.sub(r'[\[<(]?[ \t]*\w*_suppressed[ \t]*[\]>)]?', ' ', text, flags=re.IGNORECASE)


def sub_multiple_suppressed_pattern(text):  # TODO: to be tested
    processed_text = re.sub(r'.{0,1}\w*_suppressed.{0,1}', ' OMISSIS ', text, flags=re.IGNORECASE)
    return re.sub(r'(.{0,1}OMISSIS.{0,1}){2,}', ' OMISSIS ', processed_text)


def remove_bracket_ellipses(text):
    return re.sub(r'(\[\s{0,}\.\s{0,}\.\s{0,}\.\s{0,}\])', '', text)


def remove_multiple_ellipsis(text):
    return re.sub(r'((\.\s{0,}){4,})', '', text)


def remove_non_ascii(text):
    return re.sub(r'[^\u0000-\u007E\u00A1-\u00AC\u00AE-\u01FF]', '', text)


def regex_preprocessing(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(input_directory):
        file_text = open(Path.joinpath(Path(input_directory), Path(file_name))).read()

        preprocessed_file = remove_multiple_alien_topics(file_text)
        preprocessed_file = remove_end_file(preprocessed_file)
        preprocessed_file = remove_multiple_spaces(preprocessed_file)
        preprocessed_file = remove_emphasis_added_tag(preprocessed_file)
        preprocessed_file = replace_acronyms(preprocessed_file)
        preprocessed_file = remove_redacted_tag(preprocessed_file)
        preprocessed_file = remove_references(preprocessed_file)
        preprocessed_file = remove_reported_typos(preprocessed_file)
        preprocessed_file = correct_known_typos(preprocessed_file)
        preprocessed_file = remove_bracket_ellipses(preprocessed_file)
        preprocessed_file = remove_least_frequent_square_brackets(preprocessed_file)
        preprocessed_file = remove_year_brackets(preprocessed_file)
        preprocessed_file = remove_editor_name(preprocessed_file)
        preprocessed_file = remove_suppressed_pattern(preprocessed_file)
        preprocessed_file = remove_multiple_ellipsis(preprocessed_file)
        preprocessed_file = remove_non_ascii(preprocessed_file)
        preprocessed_file = remove_multiple_new_lines(preprocessed_file)

        with open(Path.joinpath(Path(output_directory), Path(file_name)), 'w') as file:
            file.write(preprocessed_file)


def regex_preprocessing_single_file(filepath):
    file = open(filepath).read()

    preprocessed_file = remove_multiple_alien_topics(file)
    preprocessed_file = remove_end_file(preprocessed_file)
    preprocessed_file = remove_multiple_spaces(preprocessed_file)
    preprocessed_file = remove_emphasis_added_tag(preprocessed_file)
    preprocessed_file = replace_acronyms(preprocessed_file)
    preprocessed_file = remove_redacted_tag(preprocessed_file)
    preprocessed_file = remove_references(preprocessed_file)
    preprocessed_file = remove_reported_typos(preprocessed_file)
    preprocessed_file = correct_known_typos(preprocessed_file)
    preprocessed_file = remove_bracket_ellipses(preprocessed_file)
    preprocessed_file = remove_least_frequent_square_brackets(preprocessed_file)
    preprocessed_file = remove_year_brackets(preprocessed_file)
    preprocessed_file = remove_editor_name(preprocessed_file)
    preprocessed_file = remove_suppressed_pattern(preprocessed_file)
    preprocessed_file = remove_multiple_ellipsis(preprocessed_file)
    preprocessed_file = remove_non_ascii(preprocessed_file)
    preprocessed_file = remove_multiple_new_lines(preprocessed_file)

    return preprocessed_file


if __name__ == '__main__':
    # filename = '038307.txt'  # '000127.txt' #
    # dataset_to_preprocess = 'train'  # Possible values: 'train', 'test'
    # filepath = Path.joinpath(Path(Path(__file__).parent.parent),
    #                          Path(f'Dataset/task1_{dataset_to_preprocess}_files_2024/{filename}'))
    # preprocessed_file = regex_preprocessing_single_file(filepath)
    #
    # print(preprocessed_file)
    # pass
    input_directory = '../Dataset/task1_train_files_2024'
    output_directory = '../Dataset/regex_preprocessed_train'
    regex_preprocessing(input_directory, output_directory)
