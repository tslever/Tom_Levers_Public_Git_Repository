import os

paths = os.listdir()

for path in paths:

    if 'Hurricane.txt' in path:

        file = open(path, 'r')

        lines = file.readlines()

        file.close()

        for i in range(0, len(lines)):

            line = lines[i]
            
            line = line.replace("0 ", "1 ", 1)

            lines[i] = line

        file = open(path, 'w')

        file.writelines(lines)

        file.close()