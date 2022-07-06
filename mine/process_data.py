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
                if num_days in ['[Discrepancy]', '[Not Available]'] :
                    continue
                death[pid] = 0
                times[pid] = num_days
                binary[pid] = 1*(int(num_days) > threshold)
            elif curr_status == 'Dead':
                num_days = clinical_df.loc[pid]['death_days_to']
                if num_days == '[Not Available]':
                    continue
                death[pid] = 1
                times[pid] = num_days
                binary[pid] = 1*(int(num_days) > threshold)
            else:
                print(pid)
                
        labels = []
        for idx in death.keys():
            labels.append(tuple((bool(int(death[idx])), int(times[idx]))))
        dt1=np.dtype(('bool,float'))
        labels = np.array(labels, dtype=dt1)

        # print(list(binary.values()))
        return np.array(list(binary.values())), np.array(death), np.array(times), np.array(labels)