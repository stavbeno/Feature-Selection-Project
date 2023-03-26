import numpy as np
import matplotlib.pyplot as plt
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
benonie_idx = script_dir.find("/examples")
benonie_dir = script_dir[:benonie_idx]
sys.path.append(benonie_dir + '/data')
sys.path.append(benonie_dir + '/src')
from data import get_pcr_data
from e_crt import EcrtTester
from utils import get_martingale_values
from sklearn.preprocessing import StandardScaler
import pandas

if __name__ == '__main__':
    date = '_220323'
    # Parameters
    seed = int(sys.argv[1])
    for _ in range(1):
        prefix = 'experiement'
        # Output directory and filename
        data_path = '/home/shalev.shaer/created_data/' + prefix + date + '/'
        #if data_path is not None:
        #    if not os.path.exists(data_path):
        #        os.mkdir(data_path)
        out_dir = data_path + prefix + date +'/'
        out_file = out_dir + "seed" + str(seed) + ".csv"
        #if os.path.exists(out_file):
        #    print("The file: --" + out_file + "-- exists")
        # Run Experiment
        j = 0  # In this setting, the tested feature is always the first one.
        n_exp = 1  # We run the test 100 times on different realizations, in order to evaluate the power and the error.
        seed_vec = np.arange(n_exp)
        tests_list = ["power", "error"]
        results_dict = {}
        for test in tests_list:
            results_dict[test] = {
                "effective n": np.zeros((n_exp,)),
                "rejected": np.zeros((n_exp,))
                            }

        for test in tests_list:
            rejected_vec = np.zeros((n_exp,))
            for ii, seed_tmp in enumerate(seed_vec):
                np.random.seed(seed)
                X, Y, beta = get_pcr_data(test=test, n=1000)        #Was 1000
                Y = StandardScaler().fit_transform(Y)
                ecrt_tester = EcrtTester(n_init=21, j=j)  # In this simple run, almost all the input parameters are the default ones.
                results_dict[test]["rejected"][ii] = ecrt_tester.run(X, Y)
                _, results_dict[test]["effective n"][ii] = get_martingale_values(ecrt_tester.martingale_dict)
            print(f"{test}  = {np.mean(results_dict[test]['rejected'])}")

        fig, ax = plt.subplots()
        ax.hist([results_dict["power"]["effective n"][results_dict["power"]["rejected"]==True],
                results_dict["power"]["effective n"][results_dict["power"]["rejected"]==False]],
                color=["limegreen", "gray"], edgecolor="black", stacked=True,
                label=["Rejected", "Not rejected"])
        plt.title("Histogram of the stopping times")
        ax.legend()
        plt.show()
        sys.stdout.flush()
        #if not os.path.exists(out_dir):
        #    os.mkdir(out_dir)
        df = pandas.DataFrame(results_dict)
        df.to_csv(out_file, index=False, float_format='%.4f')
        print(f'Update Summary of results on\n {out_file}')
        sys.stdout.flush()
