input_file = open('ar.txt','r')
output_file = open('output.txt','w')
k = 50000
#k the number of lines you want to take from the huge file
for lines in range(50000):
    line = input_file.readline()
    output_file.write(line)
    
output_file = open('output.txt','r')