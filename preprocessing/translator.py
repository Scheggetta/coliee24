import os

from lingua import Language, LanguageDetectorBuilder
from argostranslate import package, translate

from parameters import *

package.install_from_path('fr_en.argosmodel')


# TODO: preprocessing pipeline:
#  - initial regex preprocessing
#  - sentence splitting and reformatting
#  - sentence translation


def compare_french_english_script(path_to_file, output_path):
    languages = [Language.ENGLISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.4).build()
    installed_languages = translate.get_installed_languages()
    translation_fr_en = installed_languages[1].get_translation(installed_languages[0])
    text = ''
    with open(path_to_file, 'r', encoding='utf-8') as f:
        text = f.read()
        french_to_translate = detector.detect_multiple_languages_of(text)
        for lan in french_to_translate:
            if lan.language == Language.FRENCH:
                french_text = text[lan.start_index:lan.end_index]
                translated_text = translation_fr_en.translate(french_text)
                if translated_text != french_text:
                    text = text[:lan.start_index] + translated_text + text[lan.end_index:]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    folder = 'Dataset/task1_%s_files_2024' % PREPROCESSING_DATASET_TYPE
    output_folder = 'Dataset/translated_%s' % PREPROCESSING_DATASET_TYPE
    os.makedirs(output_folder, exist_ok=True)

    for idx, file_name in enumerate(os.listdir(folder)):
        compare_french_english_script(os.path.join(folder, file_name), os.path.join(output_folder, file_name))
        if (idx + 1) % 100 == 0:
            print(f'Processed {idx+1} files')
