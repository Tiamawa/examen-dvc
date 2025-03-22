import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import os


@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data (saved in ../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making intermediate data set from raw data')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))

    input_filepath_raw = f"{input_filepath}/raw.csv"

    output_filepath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())

    process_data(input_filepath_raw, output_filepath)



def process_data(input_filepath, output_filepath):
    """ Here we process input data : 
         - split data to train and test sets
         - save the results into path for further normalization.
    """

    # Import datasets
    fields = ['ave_flot_air_flow', 'ave_flot_level', 'iron_feed', 'starch_flow', 'amina_flow', 'ore_pulp_flow', 'ore_pulp_pH', 'ore_pulp_density', 'silica_concentrate']
    df = import_dataset(input_filepath, sep=",", skipinitialspace=True, usecols=fields)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to the respective output path
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    features = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    main()
