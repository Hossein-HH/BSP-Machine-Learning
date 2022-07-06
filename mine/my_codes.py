import numpy as np
import pandas as pd
import csv
import torch


# ?preprocessing
def load_clinical(patients):
    threshold = 365
    binary = {}
    death = {}
    times = {}
    clinical_df = patients
    itr = 0
    for pid in patients.index:
        itr += 1
        # assert pid not in clinical_df.index, f"Invalid Patient ID <{pid}>"
        curr_status = clinical_df.loc[pid]['vital_status']
        num_days = 0
        if curr_status == 'Alive':
            num_days = clinical_df.loc[pid]['last_contact_days_to']
            if num_days in ['[Discrepancy]', '[Not Available]']:
                continue
            death[pid] = 0
            times[pid] = num_days
            # times[pid] = int(num_days) / 365
            binary[pid] = 1 * (int(num_days) > threshold)
        elif curr_status == 'Dead':
            num_days = clinical_df.loc[pid]['death_days_to']
            if num_days == '[Not Available]':
                continue
            death[pid] = 1
            times[pid] = num_days
            # times[pid] = int(num_days) / 365
            binary[pid] = 1 * (int(num_days) > threshold)
        else:
            print(pid)

    labels = []
    for idx in death.keys():
        labels.append(tuple((bool(int(death[idx])), int(times[idx]))))
    # dt1 = np.dtype(('bool,float'))
    dt1 = np.dtype(('bool,int'))
    labels = np.array(labels, dtype=dt1)

    # print(list(binary.values()))
    return np.array(list(binary.values())), np.array(death), np.array(times), np.array(labels)


if __name__ == '__main__':
    data = pd.read_csv('clinical_with_header.csv')
    print(data["vital_status"].value_counts())
    # print data headers and types of columns in dataframe (data)
    print(data.columns.values)



    # x = load_clinical(data)
    # print("x[0] :")
    # print(x[0])
    # print("death or live array :")
    # deathOrLive = x[1]
    # print(deathOrLive)
    # print("time of live per day :")
    # timeOfLive = x[2]
    # print(timeOfLive)
    # print("x[3] :")
    # print(x[3])