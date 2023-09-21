import matplotlib.pyplot as plt
import zipfile
from datetime import datetime

from utils import read_from_zip, read_acc_file, read_wrist_file


def plot_empatica_data(zipfile, filename, ax):
    if 'IBI' in filename:
        data = read_from_zip(zipfile, filename, column_names=['time', 'duration'])
        times = [datetime.utcfromtimestamp(data['time'][i]) for i in range(len(data))]
        ax.plot(times, data['duration'])
        ax.set_title('IBI Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Seconds')
    elif 'ACC' in filename:
        times, data, _, _ = read_acc_file(zipfile, filename)
        ax.plot(times, data.iloc[:, 0], label='x')
        ax.plot(times, data.iloc[:, 1], label='y')
        ax.plot(times, data.iloc[:, 2], label='z')
        ax.legend()
        ax.set_title('ACC Data')
        ax.set_ylabel('1/64g')

    else:
        times, data, init_time, sample_rate = read_wrist_file(zipfile, filename)

        if 'TEMP' in filename:
            ax.plot(times, data.iloc[:, 0])
            ax.set_title('Temperature Data')
            ax.set_ylabel('Celsius (°C)')
        elif 'EDA' in filename:
            ax.plot(times, data.iloc[:, 0])
            ax.set_title('EDA Data')
            ax.set_ylabel('Microsiemens (μS)')
        elif 'BVP' in filename:
            ax.plot(times, data.iloc[:, 0])
            ax.set_title('BVP Data')
        elif 'HR' in filename:
            ax.plot(times, data.iloc[:, 0])
            ax.set_title('Heart Rate Data')
            ax.set_ylabel('Beats Per Minute')

        ax.set_xlabel('Time')

    mean_val = data.mean().mean()
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5)
    ax.text(0.02, 0.85, f'Mean: {mean_val:.2f}', transform=ax.transAxes, color='red', fontsize=9, fontweight='bold')

    ax.grid(True)


def main(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as z:
        filenames = [filename for filename in z.namelist() if filename.endswith('.csv') and filename != "tags.csv"]
        filenames = [filename for filename in filenames if filename != "IBI.csv"]

        # Creating subplots
        fig, axs = plt.subplots(len(filenames), 1, figsize=(10, 2.5 * len(filenames)))

        for idx, filename in enumerate(filenames):
            plot_empatica_data(z, filename, axs[idx])

    plt.tight_layout()

    output_filename = "empatica_plots.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot as {output_filename}")


if __name__ == "__main__":
    zip_filename = 'data/input/input.zip'
    main(zip_filename)
