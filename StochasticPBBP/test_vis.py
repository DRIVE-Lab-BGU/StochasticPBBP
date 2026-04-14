from StochasticPBBP.utils.helper import plot_csv_curves_from_folder
from pathlib import Path
import os


PACKAGE_ROOT = Path(__file__).resolve().parent


def main():
    domain = 'reservoir'
    instance = 2
    output_fol = os.path.join(PACKAGE_ROOT, 'outputs', domain + '_' + str(instance), 'run_logs')
    output_plot = os.path.join(output_fol, 'vis')
    plot_csv_curves_from_folder(output_fol, output_file=output_plot)


if __name__ == '__main__':
    main()