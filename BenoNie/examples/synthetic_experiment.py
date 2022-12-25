import numpy as np
import matplotlib.pyplot as plt
from data.data import get_pcr_data
from src.e_crt import EcrtTester
from src.utils import get_martingale_values

j = 0  # In this setting, the tested feature is always the first one.
n_exp = 10  # We run the test 100 times on different realizations, in order to evaluate the power and the error.
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
    for ii, seed in enumerate(seed_vec):
        np.random.seed(seed)
        X, Y, beta = get_pcr_data(test=test, n=100)        #Was 1000
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
