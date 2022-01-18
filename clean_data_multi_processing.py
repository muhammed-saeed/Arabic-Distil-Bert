import numpy as np
import io
import os
import glob

from langdetect import detect
import time
import re
import multiprocessing as mp


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F6FF"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001F900-\U0001F9FF"
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


print("script execution begins")


processed_file = open("/home/muhammed/Documents/theory_processed.txt", "a+")
counter2 = 0


def clean_line(row):

    start_time = time.time()

    initialized = True
    counter = 0
    User = ''

    text_array = row
    text_array = deEmojify(text_array)
    text_array = text_array.replace('#', '')
    text_array = text_array.split()
    filtered_text_array = []
    # if counter > 490:
    #      break
    for word in text_array:
        try:
            lang = detect(word)
        #  print(lang)
            if lang == 'ar' or lang == 'fa' or lang == 'ur':
                #  print(word + '   :  added')
                filtered_text_array.append(word)
        #  else:
        #  print(word + '   :  deleted')
        except Exception:
            #  print("#"+word + '   :  deleted')
            pass
    new_text_array = " ".join(filtered_text_array)

    # if (initialized == True):
    #   new_list = [[row['Time'] , new_text_array , row['User']]]
    #   initialized == False
    # else:

    if new_text_array != '':
        processed_file.write(new_text_array)
    # print(len(new_list))

    # print('saving file :    cleaning time: ' +
    #       str(time.time() - start_time))
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')


def launch(path):
    start_time = time.time()
    file = open("/home/muhammed/Documents/theory_4.txt", "r")
    lines = file.readlines()
    output_ = list(map(clean_line, lines))
    end_time = time.time()
    print(f"the files is cleaned and it tooks {end_time - start_time}")


if __name__ == '__main__':

    # Note
    # open python script
    # then import multiprocessing as mp
    # then check the total number of processers you have in your machine by writing the following code
    # first mp.cpu_count()
    # make sure num_processes is less than mp.cpu_count() else you will be running multithreading and this will reduce the speed of multiprocessing
    #counter2 = 0
    num_processes = 10
    # processes = []
    # # for rank in range(num_processes):
    pool = mp.Pool(num_processes)
    # pool.map(launch, "/home/muhammed/Documents/theory_4.txt")
    file = open("/home/muhammed/Documents/theory_4.txt", "r")

    lines_for_multiprocess = file.readlines()
    start_time = time.time()
    pool.map(clean_line, lines_for_multiprocess)
    end_time = time.time()
    print(f"the files is cleaned and it tooks {end_time - start_time}")

   # file_name = csvfile.split('.')
   # file_name = file_name[0]
   # file_name = file_name.split('/')
   # file_name = file_name[len(file_name) - 1]
   # print('saving file : ' + file_name)
