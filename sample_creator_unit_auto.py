'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import gc
import argparse
import os
import numpy as np
import random


from utils.data_preparation_unit import df_all_creator, df_train_creator, df_test_creator, Input_Gen


seed = 0
random.seed(0)
np.random.seed(seed)


current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')


def main():
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=10, help='window length', required=True)
    parser.add_argument('-s', type=int, default=10, help='stride of window')
    parser.add_argument('--sampling', type=int, default=1, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('--test', type=int, default='non', help='select train or test, if it is zero, then extract samples from the engines used for training')

    args = parser.parse_args()

    sequence_length = args.w
    stride = args.s
    sampling = args.sampling
    selector = args.test

    df_all = df_all_creator(data_filepath, sampling)
    # Optional: Force df_all columns to be strings here if the issue originates very early
    # if not df_all.empty:
    #     print(f"df_all original columns (first 5 types): {[type(c) for c in df_all.columns[:5]]}")
    #     df_all.columns = [str(col) for col in df_all.columns]
    #     if not all(isinstance(col, str) for col in df_all.columns):
    #         raise TypeError("CRITICAL: Not all df_all columns are Python strings after forced conversion!")
    #     print(f"df_all new columns (first 5 types): {[type(c) for c in df_all.columns[:5]]}")


    units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
    units_index_test = [11.0, 14.0, 15.0]

    print("units_index_train", units_index_train)
    print("units_index_test", units_index_test)

    df_train = df_train_creator(df_all, units_index_train)
    if not df_train.empty:
        print(f"df_train original columns (first 5 types): {[type(c) for c in df_train.columns[:5]]}")
        df_train.columns = [str(col) for col in df_train.columns] # Force conversion
        print(f"df_train new columns (first 5 types, after [str(col) ...]): {[type(c) for c in df_train.columns[:5]]}")
        if not all(isinstance(col, str) for col in df_train.columns):
            print("WARNING: Not all df_train columns are Python strings after [str(col) for col in ...]:")
            for i, col_name in enumerate(df_train.columns):
                if not isinstance(col_name, str): print(f"  df_train.columns[{i}] = '{col_name}' (type: {type(col_name)})")
            # raise TypeError("Not all df_train columns are Python strings after conversion!") # Make it a warning for now
    print(df_train)
    if not df_train.empty: print(df_train.columns) # Check output
    print("num of inputs (df_train): ", len(df_train.columns) if not df_train.empty else 0)


    df_test = df_test_creator(df_all, units_index_test)
    if not df_test.empty:
        print(f"df_test original columns (first 5 types): {[type(c) for c in df_test.columns[:5]]}")
        df_test.columns = [str(col) for col in df_test.columns] # Force conversion
        print(f"df_test new columns (first 5 types, after [str(col) ...]): {[type(c) for c in df_test.columns[:5]]}")
        if not all(isinstance(col, str) for col in df_test.columns):
            print("WARNING: Not all df_test columns are Python strings after [str(col) for col in ...]:")
            for i, col_name in enumerate(df_test.columns):
                if not isinstance(col_name, str): print(f"  df_test.columns[{i}] = '{col_name}' (type: {type(col_name)})")
            # raise TypeError("Not all df_test columns are Python strings after conversion!")
    print(df_test)
    if not df_test.empty: print(df_test.columns) # Check output
    print("num of inputs (df_test): ", len(df_test.columns) if not df_test.empty else 0)

    del df_all
    gc.collect()
    # df_all = pd.DataFrame() # Redundant

    sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
    sample_folder = os.path.isdir(sample_dir_path)
    if not sample_folder:
        os.makedirs(sample_dir_path)
        print("created folder : ", sample_dir_path)

    # Initial definitions - will be refined inside the if/else block
    # These are not strictly necessary here if defined properly inside the conditional.
    # cols_normalize = []
    # sequence_cols = []

    if (df_train.empty and selector == 0) or \
       (df_test.empty and selector != 0 and df_train.empty): # Need df_train for scaler even in test mode
        print("Warning: Required DataFrame is empty for the current selector. Skipping Input_Gen.")
    else:
        if selector == 0:
            if df_train.empty: # Should be caught by the outer check, but good for safety
                print("Skipping training data generation as df_train is empty.")
            else:
                # df_train.columns are now (hopefully) Python strings
                _cols_to_exclude = ['RUL', 'unit'] # Ensure these are Python strings

                cols_normalize = [col for col in df_train.columns if col not in _cols_to_exclude]
                sequence_cols = [col for col in df_train.columns if col not in _cols_to_exclude] # Adjust if different

                print(f"cols_normalize (train path, first 5): {cols_normalize[:5]}")
                print(f"Types in cols_normalize (train path, first 5): {[type(c) for c in cols_normalize[:5]]}")

                # Critical check for the slice being passed to the scaler
                df_slice_for_scaler = df_train[cols_normalize]
                print(f"Columns of df_slice_for_scaler (train path, first 5 types): {[type(c) for c in df_slice_for_scaler.columns[:5]]}")
                if not all(isinstance(col, str) for col in df_slice_for_scaler.columns):
                    print("CRITICAL WARNING (TRAIN PATH): Slice columns for scaler are not all Python strings!")
                    for i, col_name in enumerate(df_slice_for_scaler.columns):
                        if not isinstance(col_name, str): print(f"  df_slice_for_scaler.columns[{i}] = '{col_name}' (type: {type(col_name)})")


                print("the number of input signals: ", len(cols_normalize))
                for unit_index in units_index_train:
                    data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                            unit_index, sampling, stride =stride)
                    data_class.seq_gen()
        else: # selector != 0 (test data generation)
            if df_train.empty: # Scaler is fit on df_train
                 print("Skipping test data generation as df_train (for scaler fitting) is empty.")
            elif df_test.empty:
                 print("Skipping test data generation as df_test is empty.")
            else:
                _cols_to_exclude = ['RUL', 'unit']

                # cols_normalize uses df_train because the scaler is fit on df_train
                cols_normalize = [col for col in df_train.columns if col not in _cols_to_exclude]
                # sequence_cols_test should be derived from df_test columns
                sequence_cols_test = [col for col in df_test.columns if col not in _cols_to_exclude]

                print(f"cols_normalize (test path, from df_train, first 5): {cols_normalize[:5]}")
                print(f"Types in cols_normalize (test path, first 5): {[type(c) for c in cols_normalize[:5]]}")
                print(f"sequence_cols_test (test path, from df_test, first 5): {sequence_cols_test[:5]}")
                print(f"Types in sequence_cols_test (test path, first 5): {[type(c) for c in sequence_cols_test[:5]]}")

                # Critical check for the slice of df_train being passed to the scaler
                df_slice_for_scaler = df_train[cols_normalize]
                print(f"Columns of df_slice_for_scaler (test path, from df_train, first 5 types): {[type(c) for c in df_slice_for_scaler.columns[:5]]}")
                if not all(isinstance(col, str) for col in df_slice_for_scaler.columns):
                     print("CRITICAL WARNING (TEST PATH): df_train slice columns for scaler are not all Python strings!")
                     for i, col_name in enumerate(df_slice_for_scaler.columns):
                        if not isinstance(col_name, str): print(f"  df_slice_for_scaler.columns[{i}] = '{col_name}' (type: {type(col_name)})")


                print("the number of input signals (for test processing, based on train): ", len(cols_normalize))
                for unit_index in units_index_test:
                    data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols_test, sample_dir_path,
                                            unit_index, sampling, stride =stride)
                    data_class.seq_gen()

if __name__ == '__main__':
    main()
