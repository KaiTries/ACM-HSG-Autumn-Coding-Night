import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def read_transaction_data(DIR_INPUT, BEGIN_DATE, END_DATE):
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if
             f >= BEGIN_DATE + '.pkl' and f <= END_DATE + '.pkl']

    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)

    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)
    #  Note: -1 are missing values for realpi world data
    df_final = df_final.replace([-1], 0)

    # drop the TX_DATETIME column because ai model only works with numbers
    df_final["TX_DATETIME"] = df_final["TX_DATETIME"].astype(np.int64)
    return df_final

def write_list_to_text_file(filename, input_list):
    try:
        with open(filename, 'w') as file:
            for item in input_list:
                file.write(str(item) + '\n')
        print(f"Successfully wrote {len(input_list)} items to {filename}")
    except IOError as e:
        print(f"Error writing to {filename}: {e}")

DIR_INPUT='../simulated-training-data-raw'

BEGIN_DATE = "2018-04-01"
END_DATE = "2018-09-30"

print("Load  files...")
transactions_df=read_transaction_data(DIR_INPUT, BEGIN_DATE, END_DATE)
print("Done")
train, test = train_test_split(transactions_df, test_size=0.2)
X_train = train[["TX_DATETIME","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_TIME_SECONDS","TX_TIME_DAYS"]]
Y_train = train["TX_FRAUD"]

X_test = test[["TX_DATETIME","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_TIME_SECONDS","TX_TIME_DAYS"]]
Y_test = test["TX_FRAUD"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
X_test = scaler.fit_transform(X_test)
# try model out for other data
predictions = knn.predict(X_test)
print("Accuracy: ", accuracy_score(Y_test, predictions))


actual_input = read_transaction_data("../training-data/input", "2018-04-01", "2018-09-30")
actual_input_data = actual_input[["TX_DATETIME","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_TIME_SECONDS","TX_TIME_DAYS"]]
X_data = scaler.fit_transform(actual_input_data)
predictions = knn.predict(X_data)

actual_input["prediction"] = predictions

filtered_df = actual_input[actual_input['prediction'] == 1]

ids = filtered_df["TRANSACTION_ID"]

lst = ids.to_numpy()

lst = lst.reshape(1,-1)

lst = lst.tolist()

write_list_to_text_file("predictions.txt", lst[0])