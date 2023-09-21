import pandas as pd
from io import BytesIO
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def read_from_zip(zipfile, filename, column_names=None):
    with zipfile.open(filename) as file:
        return pd.read_csv(BytesIO(file.read()), header=None, names=column_names)


def read_acc_file(zipfile, filename):
    data = read_from_zip(zipfile, filename, column_names=['ACC_x', 'ACC_y', 'ACC_z'])
    init_time = data.iloc[0, 0]
    sample_rate = data.iloc[1, 0]
    data = data.iloc[2:]
    times = [init_time + i / sample_rate for i in range(data.shape[0])]
    return times, data, init_time, sample_rate


def read_wrist_file(zipfile, filename, column_name='data'):
    data = read_from_zip(zipfile, filename, column_names=[column_name])
    init_time = data.iloc[0, 0]
    sample_rate = data.iloc[1, 0]
    data = data.iloc[2:]
    times = [init_time + i / sample_rate for i in range(data.shape[0])]
    return times, data, init_time, sample_rate


def show_and_save_predicitons(df_preds, start_datetime):
    start = datetime.utcfromtimestamp(start_datetime)
    date_range = [start + timedelta(seconds=i) for i in range(len(df_preds))]
    df_preds['date'] = date_range

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))

    # Line plot
    ax.plot(date_range, df_preds['labels'], linestyle='-', color='black')

    # Setting yticks to only show 0 and 1
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_ylim(-1, 2)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_title('Model predictions')
    ax.set_xlabel('Time (HH:MM:SS) (UTC+3)')
    ax.set_ylabel('Value')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # Rotate x labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('predictions/predictions-graph.png', dpi=300)
    plt.show()

    df_preds['change'] = (df_preds['labels'] != df_preds['labels'].shift()).astype(int)
    df_preds['group'] = df_preds['change'].cumsum()

    # Grouping by the sequences and extracting start and end times
    result = df_preds.groupby(['labels', 'group'])['date'].agg(['min', 'max']).reset_index()

    # Filtering for periods with label = 1 (stress periods)
    stress_periods = result[result['labels'] == 1].drop('labels', axis=1)

    # Save to text file
    with open("predictions/stress_periods.txt", "w") as file:
        for _, row in stress_periods.iterrows():
            file.write(f"For stress: From {row['min']} To {row['max']}\n")

    print(stress_periods)
