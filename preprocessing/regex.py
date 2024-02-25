from pathlib import Path
import os
import re


# TODO:
#  - get the year of the document
#  - assess what's inside the [] tags
#  - preprocessing to remove useless new lines, <FRAGMENT_SUPPRESSES>, etc...
#  - there are other keywords in the text that should be removed, like <FRAGMENT_SUPPRESSES>, [DATE_SUPPRESSED], `_SUPPRESSED`, REFERENCE_SUPPRESSED, CITATION_SUPPRESSED, etc...
#  - there are other non visible characters that should be removed, like  
#  - remove tab/space at the beginning of the line
#  - remove Editor's note, like `Editor: Marco Rossi`
#  - remove frequent notes, like `[End of document]`, `[Translation]`, `MLB headnote and full text`, `This case is unedited, therefore contains no summary.`, `[French language version follows English language version]`, `[La version française vient à la suite de la version anglaise]`, `MLB unedited judgment`, etc...


def remove_multiple_new_lines(text):
    processed_file = re.sub(r'(\n{2,})|(\s\n){2,}', '\n', text)
    processed_file = re.sub(r'(\n\s)', '\n', processed_file)
    if processed_file[0] == '\n':
        processed_file = processed_file[1:]
    return processed_file


def remove_end_file(text):
    return re.sub(r'\[End of document\]', '', text)


def remove_editor_name(text):
    return re.sub(r'Editor:.*\n', '', text)


def remove_suppressed_pattern(text):
    return re.sub(r'.{0,1}\w*_suppressed.{0,1}', '', text, flags=re.IGNORECASE)


def sub_multiple_suppressed_pattern(text):  # TODO: to be tested
    processed_text = re.sub(r'.{0,1}\w*_suppressed.{0,1}', ' OMISSIS ', text, flags=re.IGNORECASE)
    return re.sub(r'(.{0,1}OMISSIS.{0,1}){2,}', ' OMISSIS ', processed_text)


def remove_multiple_ellipsis(text):
    return re.sub(r'((\.\s{0,}){4,})', '', text)


def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)


def regex_preprocessing(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(input_directory):
        file_text = open(Path.joinpath(Path(input_directory), Path(file_name))).read()

        # TODO: refactor with a single function
        processed_file = remove_end_file(file_text)
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
    preprocessed_file = remove_editor_name(preprocessed_file)
    preprocessed_file = remove_suppressed_pattern(preprocessed_file)
    preprocessed_file = remove_multiple_ellipsis(preprocessed_file)
    preprocessed_file = remove_non_ascii(preprocessed_file)
    preprocessed_file = remove_multiple_new_lines(preprocessed_file)

    return preprocessed_file


if __name__ == '__main__':
    filepath = Path.joinpath(Path(Path(__file__).parent.parent), Path('Dataset/task1_train_files_2024/000002.txt'))
    preprocessed_file = regex_preprocessing_single_file(filepath)

    print(preprocessed_file)
    pass
