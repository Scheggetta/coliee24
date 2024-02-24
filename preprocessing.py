from openai import OpenAI
import re
import os
from pathlib import Path
import nltk


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


directory = 'Dataset/Train_Queries'
file_name = '001153.txt'  # os.listdir(directory)[3]
file_text = open(Path.joinpath(Path(directory), Path(file_name))).read()

processed_file = remove_end_file(file_text)
processed_file = remove_editor_name(processed_file)
processed_file = remove_suppressed_pattern(processed_file)  # if switch else sub_multiple_suppressed_pattern(processed_file)
processed_file = remove_multiple_ellipsis(processed_file)
processed_file = remove_non_ascii(processed_file)
processed_file = remove_multiple_new_lines(processed_file)
quit(0)

client = OpenAI()

MAX_TOKENS_INPUT = 3900

# TODO:
#  - get the year of the document
#  - assess what's inside the [] tags
#  - preprocessing to remove useless new lines, <FRAGMENT_SUPPRESSES>, etc...
#  - there are other keywords in the text that should be removed, like <FRAGMENT_SUPPRESSES>, [DATE_SUPPRESSED], `_SUPPRESSED`, REFERENCE_SUPPRESSED, CITATION_SUPPRESSED, etc...
#  - there are other non visible characters that should be removed, like  
#  - remove tab/space at the beginning of the line
#  - remove Editor's note, like `Editor: Marco Rossi`
#  - remove frequent notes, like `[End of document]`, `[Translation]`, `MLB headnote and full text`, `This case is unedited, therefore contains no summary.`, `[French language version follows English language version]`, `[La version française vient à la suite de la version anglaise]`, `MLB unedited judgment`, etc...

file = open("/home/edo/PycharmProjects/coliee24/Dataset/Train_Queries/001299.txt").read()

# Split the text into paragraphs using re library.
# Each paragraph starts with `[N]` where N is an integer number with maximum 3 digits
text = re.split(r'\[\d{1,3}\]', file)

# TODO:
#  - tokenization to count how many tokens to give to the model for each message
#  - parallelize model calls


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
