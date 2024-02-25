import os
from pathlib import Path
import re

from openai import OpenAI

from preprocessing.regex import regex_preprocessing, regex_preprocessing_single_file


dataset_to_preprocess = 'test'  # Possible values: 'train', 'test'
TRAIN_DATASET_DIR = Path.joinpath(Path(Path(__file__).parent.parent), Path('Dataset/task1_train_files_2024'))
TEST_DATASET_DIR = Path.joinpath(Path(Path(__file__).parent.parent), Path('Dataset/task1_test_files_2024'))
DATASET_DIR = TRAIN_DATASET_DIR if dataset_to_preprocess == 'train' else TEST_DATASET_DIR
REGEX_PREPROCESSED_DIR = Path.joinpath(Path(Path(__file__).parent.parent),
                                       Path('Dataset/regex_preprocessed_%s' % dataset_to_preprocess))

MAX_TOKENS_INPUT = 3900


if not Path.exists(REGEX_PREPROCESSED_DIR):
    regex_preprocessing(input_directory=DATASET_DIR, output_directory=REGEX_PREPROCESSED_DIR)
# Get regex preprocessed files from the file system
regex_preprocessed_files = [open(Path.joinpath(REGEX_PREPROCESSED_DIR, Path(f))).read() for f in os.listdir(REGEX_PREPROCESSED_DIR)]


# Split the text into paragraphs using re library.
# Each paragraph starts with `[N]` where N is an integer number with maximum 3 digits
preprocessed_file = regex_preprocessed_files[0]

# preprocessed_file = open(Path.joinpath(REGEX_PREPROCESSED_DIR, Path('069639.txt'))).read()
paragraphs = re.split(r'\[\d{1,4}\]', preprocessed_file)

# TODO:
#  - tokenization to count how many tokens to give to the model for each message
#  - parallelize model calls


client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a legal case document that you have to preprocess. In particular, in the middle of many sentences there is a new line, so you have to keep the whole sentence in only one line. You have to preprocess the document WITHOUT changing any word."
    },
    {
      "role": "user",
      "content": open("/home/edo/PycharmProjects/coliee24/Dataset/Train_Evidence/094835.txt").read()
    }
  ],
  temperature=0.0,
  seed=62
)

print(response.choices[0].message.content)
print(response.system_fingerprint)
pass
