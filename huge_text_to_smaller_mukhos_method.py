lines_per_file = 55000
smallfile = None
num_lines = sum(1 for line in open('ar.txt'))
with open('ar.txt') as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = '/home/arabic-distillbert/araDistilBertVocabFile2/small_file_{}.txt'.format(lineno + lines_per_file)
            smallfile = open(small_filename, "w")
            print(str(lineno) + '/' + num_lines)
        smallfile.write(line)
    if smallfile:
        smallfile.close()