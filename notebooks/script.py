import pandas as pd
import os
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

    return df_final

def write_list_to_text_file(filename, input_list):
    try:
        with open(filename, 'w') as file:
            for item in input_list:
                file.write(item + '\n')
        print(f"Successfully wrote {len(input_list)} items to {filename}")
    except IOError as e:
        print(f"Error writing to {filename}: {e}")

# Load data from the 2018-07-25 to the 2018-08-14

DIR_INPUT='../simulated-training-data-raw'

BEGIN_DATE = "2018-07-25"
END_DATE = "2018-08-14"

print("Load  files...")
transactions_df=read_transaction_data(DIR_INPUT, BEGIN_DATE, END_DATE)
print("Done")


transactions_df["TX_DATETIME"] = transactions_df["TX_DATETIME"].astype(np.int64)

train, test = train_test_split(transactions_df, test_size=0.2)

X_train = train[["TRANSACTION_ID","TX_DATETIME","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_TIME_SECONDS","TX_TIME_DAYS"]]

X_test = test[["TRANSACTION_ID","TX_DATETIME","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_TIME_SECONDS","TX_TIME_DAYS"]]

Y_train = train["TX_FRAUD"]

Y_test = test["TX_FRAUD"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# Load data from the 2018-07-25 to the 2018-08-14

DIR_INPUT='../training-data/input'

BEGIN_DATE = "2018-04-01"
END_DATE = "2018-09-30"

print("Load  files...")
input_training=read_transaction_data(DIR_INPUT, BEGIN_DATE, END_DATE)
print("Done")

input_training["TX_DATETIME"] = input_training["TX_DATETIME"].astype(np.int64)

input_training_data = scaler.fit_transform(input_training)


predictions = knn.predict(input_training)
# count errors
errors = 0
count = 0
output = []

for index, item in enumerate(predictions):
    if item == 1:
        count += 1
        output.append(str(int(input_training.iloc[index]["TRANSACTION_ID"])))
        if df1.iloc[index]["TX_FRAUD"] == 0:
            errors += 1
    else:
        if df1.iloc[index]["TX_FRAUD"] == 1:
            errors += 1

print("Predicted Frauds: ", count)
print("Actual Frauds: ", df1["TX_FRAUD"].value_counts().get(1, 0))

print("Errors: ", errors)

write_list_to_text_file("output.txt", output)