import os
import data_features_preparation
import warnings
import pandas as pd


from utils import show_and_save_predicitons
from wesad_training import do_training, plot_accuracies_losses, do_confusion_matrix, calculate_f1_score, \
    read_and_prepare_data, load_and_predict, load_predictions_from_csv

if __name__ == "__main__":
    training_part = False
    warnings.filterwarnings('ignore')

    if training_part:
        wesad_base_directory = 'data/WESAD/'
        dirnames = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

        for dirname in dirnames:
            full_path = os.path.join(wesad_base_directory, dirname)
            print(f'Processing data for {dirname}...')
            data_features_preparation.extract_features(full_path, dirname, train=True)

        data_features_preparation.combine_files(wesad_base_directory, dirnames)

        subject_id_list, df = read_and_prepare_data()

        y_truths, y_preds, test_accs, test_losses = do_training(subject_id_list, df)

        plot_accuracies_losses(test_accs, test_losses, subject_id_list)
        do_confusion_matrix(y_truths, y_preds, subject_id_list)
        calculate_f1_score(y_truths, y_preds, subject_id_list)

    else:
        df, eda_df, bvp_df, temp_df, _, input_df, start_datetime = data_features_preparation.extract_features(
            'data/input/',
            dirname=None,
            train=False)

        predictions = load_and_predict(input_df)
        df_preds = pd.DataFrame(predictions, columns=['labels'])
        df_preds = pd.DataFrame(df_preds['labels'].repeat((
            data_features_preparation.fs_dict['label'] * data_features_preparation.WINDOW_IN_SECONDS) / data_features_preparation.fs_dict['BVP']).reset_index(drop=True))

        show_and_save_predicitons(df_preds, start_datetime)
