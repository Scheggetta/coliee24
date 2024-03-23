import os
import multiprocessing

from lingua import Language, LanguageDetectorBuilder
from argostranslate import package, translate

from parameters import *

package.install_from_path('fr_en.argosmodel')


def translate_file(filepath, detector, translator):
    translation = ''

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        sentences = text.split('\n')

        confidences = detector.compute_language_confidence_in_parallel(sentences, Language.FRENCH)
        for sentence, confidence in zip(sentences, confidences):
            if confidence >= SEUIL:
                translation += translator.translate(sentence) + '\n'
            else:
                translation += sentence + '\n'
    return translation


def translate_folder(input_filenames, input_folder, output_folder, detector, translator):
    os.makedirs(output_folder, exist_ok=True)

    for idx, file_name in enumerate(input_filenames):
        input_filepath = os.path.join(input_folder, file_name)
        output_filepath = os.path.join(output_folder, file_name)

        translated_file = translate_file(input_filepath, detector=detector, translator=translator)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(translated_file)

        if (idx + 1) % 100 == 0:
            print(f'Processed {idx + 1} files')


if __name__ == '__main__':
    folder = 'Dataset/sentence_preprocessing_%s' % PREPROCESSING_DATASET_TYPE
    output_folder = 'Dataset/translated_%s' % PREPROCESSING_DATASET_TYPE
    os.makedirs(output_folder, exist_ok=True)

    languages = [Language.ENGLISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    installed_languages = translate.get_installed_languages()
    translator = installed_languages[1].get_translation(installed_languages[0])

    # filename = '003523.txt'
    # filepath = os.path.join(folder, filename)
    # translated_file = translate_file(filepath, detector=detector, translator=translator)
    # quit()

    n_processes = 15
    filenames = os.listdir(folder)
    split_size = len(filenames) // n_processes
    split_filenames = [filenames[i * split_size:(i + 1) * split_size] for i in range(n_processes + 1)]

    translate_folder(os.listdir(folder), folder, output_folder, detector, translator)
