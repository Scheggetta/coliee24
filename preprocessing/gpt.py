import os
from pathlib import Path
import re
import time
import concurrent.futures

from openai import OpenAI
import tiktoken
import pickle
from preprocessing.regular_exp import regex_preprocessing, regex_preprocessing_single_file


SYSTEM_PROMPT = "You will be provided with a legal case document that you have to preprocess.\n\nAt the very beginning of the document there could be a part where the name of the case, applicants, respondent, dates, counsels, solicitors of records, references to other cases, topics, notes, summaries and related information are present. Out of all this, KEEP ONLY the main summary; then, you can continue the normal preprocessing.\n\nWhat you must do after that initial phase:\n- keep the whole sentence in the same line;\n- add new line characters based on the paragraphs' contents;\n- do NOT change any word;\n- do NOT remove any word;\n- do NOT add any word, just new line characters if necessary."

encoding = tiktoken.get_encoding("cl100k_base")


def get_token_count(paragraphs):
    token_count = []
    for paragraph in paragraphs:
        token_count.append(len(encoding.encode(paragraph)))
    return token_count, sum(token_count)


dataset_to_preprocess = 'test'  # Possible values: 'train', 'test'
TRAIN_DATASET_DIR = Path.joinpath(Path(Path(__file__).parent.parent), Path('Dataset/task1_train_files_2024'))
TEST_DATASET_DIR = Path.joinpath(Path(Path(__file__).parent.parent), Path('Dataset/task1_test_files_2024'))
DATASET_DIR = TRAIN_DATASET_DIR if dataset_to_preprocess == 'train' else TEST_DATASET_DIR
REGEX_PREPROCESSED_DIR = Path.joinpath(Path(Path(__file__).parent.parent),
                                       Path('Dataset/regex_preprocessed_%s' % dataset_to_preprocess))
TRANSLATED_DIR = Path.joinpath(Path(Path(__file__).parent.parent),
                               Path('Dataset/translated_%s' % dataset_to_preprocess))

MAX_TOKENS_INPUT = 3900
MAX_EMBED_INPUT = 8192
GPT_MODEL = 'gpt-3.5-turbo-0125'

if not Path.exists(REGEX_PREPROCESSED_DIR):
    regex_preprocessing(input_directory=DATASET_DIR, output_directory=REGEX_PREPROCESSED_DIR)


# Get regex preprocessed files from the file system
# regex_preprocessed_files = [open(Path.joinpath(REGEX_PREPROCESSED_DIR, Path(f))).read() for f in os.listdir(REGEX_PREPROCESSED_DIR)]


def gpt_preprocessing(input_directory, output_directory):
    client = OpenAI()
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(input_directory):
        preprocessed_file = gpt_preprocessing_single_file(Path.joinpath(Path(input_directory), Path(file_name)),
                                                          client=client)

        with open(Path.joinpath(Path(output_directory), Path(file_name)), 'w') as file:
            file.write(preprocessed_file)


def gpt_preprocessing_parallelized(input_directory, output_directory):
    # client = OpenAI()
    os.makedirs(output_directory, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(gpsf,
                                          Path.joinpath(Path(input_directory), Path(file_name))
                                          ): file_name for file_name in os.listdir(input_directory)}
        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                preprocessed_file = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (file_name, exc))
            else:
                with open(Path.joinpath(Path(output_directory), Path(file_name)), 'w') as file:
                    file.write(preprocessed_file)


def gpsf(filepath):
    client = OpenAI()
    file = open(filepath).read()

    paragraphs = re.split(r'\[\d{1,4}\]', file)
    token_count, total_token_count = get_token_count(paragraphs)

    current_token_count = 0
    text = ['']
    preprocessed_file = ''
    for i, count in enumerate(token_count):
        if text[-1] == '':
            text[-1] = paragraphs[i]
            current_token_count = count
        elif current_token_count + count <= MAX_TOKENS_INPUT:
            text[-1] += paragraphs[i]
            current_token_count += count
        else:
            text.append(paragraphs[i])
            current_token_count = count

    for i, t in enumerate(text):
        print('Processing part %d of %d - file: %s' % (i + 1, len(text), str(filepath).split('/')[-1]))
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": t
                }
            ],
            temperature=0.0,
            seed=62
        )
        preprocessed_file += '\n\n' + response.choices[0].message.content
        # print(response.system_fingerprint)

    return preprocessed_file


def gpt_preprocessing_single_file(filepath, client):
    file = open(filepath).read()

    paragraphs = re.split(r'\[\d{1,4}\]', file)
    token_count, total_token_count = get_token_count(paragraphs)

    current_token_count = 0
    text = ['']
    preprocessed_file = ''
    for i, count in enumerate(token_count):
        if text[-1] == '':
            text[-1] = paragraphs[i]
            current_token_count = count
        elif current_token_count + count <= MAX_TOKENS_INPUT:
            text[-1] += paragraphs[i]
            current_token_count += count
        else:
            text.append(paragraphs[i])
            current_token_count = count

    for i, t in enumerate(text):
        print('Processing part %d of %d - file: %s' % (i + 1, len(text), str(filepath).split('/')[-1]))
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": t
                }
            ],
            temperature=0.0,
            seed=62
        )
        preprocessed_file += '\n\n' + response.choices[0].message.content
        # print(response.system_fingerprint)

    return preprocessed_file


def embed_text_gpt(filepath, file_name, output_directory):
    text = open(filepath, encoding='utf-8').read()
    paragraphs = re.split(r'\[\d{1,4}\]', text)
    token_count, total_token_count = get_token_count(paragraphs)

    current_token_count = 0
    text_calls = []
    for i, count in enumerate(token_count):
        if text_calls == [] and count <= MAX_EMBED_INPUT:
            text_calls.append(paragraphs[i])
            current_token_count = count
        elif current_token_count + count <= MAX_EMBED_INPUT:
            text_calls[-1] += paragraphs[i]
            current_token_count += count
        elif count <= MAX_EMBED_INPUT:
            text_calls.append(paragraphs[i])
            current_token_count = count
        else:
            print('Found a paragraph that exceeds the token limit (%s): %s' % (file_name, paragraphs[i]))
            iteration = 0
            while count > 0:
                curr_par = paragraphs[i][iteration * MAX_EMBED_INPUT: (iteration + 1) * MAX_EMBED_INPUT]
                text_calls.append(curr_par)
                count -= MAX_EMBED_INPUT
                iteration += 1

    client = OpenAI()
    embed = []
    for call in text_calls:
        embed += client.embeddings.create(input=[call], model='text-embedding-3-small').data[0].embedding

    os.makedirs(output_directory, exist_ok=True)
    backup_dir = Path.joinpath(Path(output_directory), 'backup')
    os.makedirs(backup_dir, exist_ok=True)

    with open(Path.joinpath(backup_dir, Path(file_name)), 'w') as file:
        for e in embed:
            file.write(str(e))
            file.write('\n')
    with open(Path.joinpath(Path(output_directory), Path(file_name)), 'wb') as file:
        pickle.dump(embed, file)


if __name__ == '__main__':
    start = time.time()

    dataset_folder = Path.joinpath(Path('..'), 'Dataset')
    input_path = TRANSLATED_DIR
    output_path = Path.joinpath(dataset_folder, 'gpt_embed_%s' % dataset_to_preprocess)

    for idx, file_name in enumerate(os.listdir(TRANSLATED_DIR)):
        filepath = Path.joinpath(input_path, Path(file_name))
        if not Path.joinpath(output_path, Path(file_name)).exists():
            if idx % 10 == 0:
                print(f'Processing {file_name}...')
            embed_text_gpt(filepath, file_name, output_path)

    # gpt_preprocessing_parallelized(input_path, output_path)

    # gpt_preprocessing_parallelized(input_directory='/home/edo/PycharmProjects/coliee24/Dataset/gpt_test',
    #                                output_directory='/home/edo/PycharmProjects/coliee24/Dataset/gpt_test_output')
    print(time.time() - start)
    print()
