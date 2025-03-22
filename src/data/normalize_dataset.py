import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import os
from sklearn.preprocessing import normalize


@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    
    logger = logging.getLogger(__name__)
    logger.info('normalizing data after splitting')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))

    input_train = f"{input_filepath}/X_train.csv"
    input_test = f"{input_filepath}/X_test.csv"

    output_filepath = click.prompt('Enter the file path for the output normalized data (e.g., output/preprocessed_data.csv)', type=click.Path())

    X_train_scaled = normalize_data(input_train)
    X_test_scaled = normalize_data(input_test)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to the respective output path
    save_dataframes(X_train_scaled, X_test_scaled, output_filepath)



def normalize_data(df):
    # Normalize given input data : feature-wise
    norm_df = normalize(df, norm='L2', axis=0)

    return norm_df


def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)


def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    main()
