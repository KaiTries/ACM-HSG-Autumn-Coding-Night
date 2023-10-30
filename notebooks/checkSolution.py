import os
import pandas as pd

# Load a set of pickle files, put them together in a single DataFrame, and order them by time
# It takes as input the folder DIR_INPUT where the files are stored, and the BEGIN_DATE and END_DATE

def read_transaction_data(DIR_INPUT):
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT)]

    frames = []

    for f in files:
        with open(f, 'r') as file:
            for line in file:
                frames.append(line.strip())
    return frames

listOfResults = read_transaction_data("../training-data/output")
def write_list_to_text_file(filename, input_list):
    try:
        with open(filename, 'w') as file:
            for item in input_list:
                file.write(item + '\n')
        print(f"Successfully wrote {len(input_list)} items to {filename}")
    except IOError as e:
        print(f"Error writing to {filename}: {e}")

write_list_to_text_file("output.txt", listOfResults)