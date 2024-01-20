import os

def save_files(path, output_file_path):
    with open(output_file_path, 'w') as output_file:
        files = os.listdir(path)
        for i in range(0, len(files), 2):
            file_name = files[i]
            full_path = os.path.join(path, file_name)
            if os.path.isfile(full_path):
                output_file.write(file_name[:-5] + '\n')

path = 'D:\zdj_satelitarne'
output_file_path = 'images.txt'

save_files(path, output_file_path)