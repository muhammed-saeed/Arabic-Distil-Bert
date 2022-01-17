from base64 import decode
from cgitb import text
import json
import io
import os
import gzip

files = list(filter(lambda x: x.endswith(".json.gz"), os.listdir()))
# list all the files those ending with json
print(files)
length_ = int(len(files) / 4)
files_to_be_processed = files[:length_]


def processing(encoded_line):
    decoded_line = encoded_line.strip()
    # arabic data_line

    return json.loads(decoded_line)


print(files_to_be_processed)
print(length_)


def pre_processing_all(my_file):

    # text_data = open("c4-ar.txt", encoding="utf-8");
    text_data = gzip.open(my_file)

    encoded_lines = text_data.readlines()

    # print(encoded_lines)

    decoded_lines = list(map(processing, encoded_lines))
    text_file = list(map(lambda x: x["text"], decoded_lines))

    textfile = open("theory_4.txt", "a+")
    for element in text_file:
        textfile.write(element + "\n")
    textfile.close()
    return my_file
    # json.dump(decoded_lines,"theory.txt");


list(map(pre_processing_all, files_to_be_processed))
