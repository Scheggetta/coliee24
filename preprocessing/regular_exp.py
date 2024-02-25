from pathlib import Path
import os
import re
from data_analyser import get_bracket_freqs_dataset


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
    return re.sub(r'(\[sic\]|\[SIC\])', '', text)


def correct_known_typos(text):
    return re.sub(r'\[([a-zA-Z])\]', r'\1', text)


def remove_least_frequent_square_brackets(text):
    freqs = get_bracket_freqs_dataset()
    for key in freqs.keys():
        if freqs[key] < 50:
            text = text.replace(key, '')
    return text


def remove_multiple_spaces(text):
    return re.sub(r'\s{2,}|\n\s{1,}|\n\t{1,}', ' ', text)


def remove_multiple_new_lines(text):
    processed_file = re.sub(r'(\n{2,})|(\s\n){2,}', '\n', text)
    processed_file = re.sub(r'(\n\s)', '\n', processed_file)
    if processed_file[0] == '\n':
        processed_file = processed_file[1:]
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


def remove_suppressed_pattern(text):
    return re.sub(r'.{0,1}\w*_suppressed.{0,1}', '', text, flags=re.IGNORECASE)


def sub_multiple_suppressed_pattern(text):  # TODO: to be tested
    processed_text = re.sub(r'.{0,1}\w*_suppressed.{0,1}', ' OMISSIS ', text, flags=re.IGNORECASE)
    return re.sub(r'(.{0,1}OMISSIS.{0,1}){2,}', ' OMISSIS ', processed_text)


def remove_bracket_ellipses(text):
    return re.sub(r'(\[\s{0,}\.\s{0,}\.\s{0,}\.\s{0,}\])', '', text)


def remove_multiple_ellipsis(text):
    return re.sub(r'((\.\s{0,}){4,})', '', text)


def remove_non_ascii(text): # FIXME: this is not working
    return re.sub(r'[^\x00-\x7F]+', '', text)


def regex_preprocessing(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(input_directory):
        file_text = open(Path.joinpath(Path(input_directory), Path(file_name))).read()

        # TODO: refactor with a single function
        processed_file = remove_end_file(file_text)
        processed_file = remove_year_brackets(processed_file)
        processed_file = remove_editor_name(processed_file)
        processed_file = remove_suppressed_pattern(processed_file)  # if switch else sub_multiple_suppressed_pattern(processed_file)
        processed_file = remove_multiple_ellipsis(processed_file)
        processed_file = remove_non_ascii(processed_file)
        processed_file = remove_multiple_new_lines(processed_file)

        with open(Path.joinpath(Path(output_directory), Path(file_name)), 'w') as file:
            file.write(processed_file)


def regex_preprocessing_single_file(filepath):
    file = open(filepath).read()

    preprocessed_file = remove_end_file(file)
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
    filename = '038307.txt'  # '000127.txt' #
    dataset_to_preprocess = 'train'  # Possible values: 'train', 'test'
    filepath = Path.joinpath(Path(Path(__file__).parent.parent),
                             Path(f'Dataset/task1_{dataset_to_preprocess}_files_2024/{filename}'))
    preprocessed_file = regex_preprocessing_single_file(filepath)

    print(preprocessed_file)
    pass
