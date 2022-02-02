import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression

local_filenames = [
    "output/out_ba_50000_5_dist_b.txt"
]


def obtain_average_distribution(filenames):
    for filename in filenames:
        f = open(filename)
        lines = f.readlines()
        less_one = 0
        over_one = 0
        dist = [0 for x in range(500)]
        maxv = 0
        for line in lines:
            data = line.split(' ')
            for i in range(len(data)):
                int_data = int(data[i])
                if i == 0:
                    less_one += int_data
                else:
                    over_one += int_data
                dist[i] += int_data
                if int_data > maxv:
                    maxv = int_data
        print(f"{round((over_one / (less_one + over_one) * 100), 3)}%")

        lines_count = len(lines)
        dist = [v / lines_count for v in dist]

        # leave only non-zero
        n_bins = zip(dist, range(len(dist)))
        n_bins = list(filter(lambda x: x[0] > 0.5, n_bins))
        n, bins = [ a for (a,b) in n_bins ], [ b for (a,b) in n_bins ]
        
        # get log-log scale distribution
        lnt, lnb = [], []
        for i in range(len(bins) - 1):
            if (n[i] != 0):
                lnt.append(math.log(bins[i]+1))
                lnb.append(math.log(n[i]) if n[i] != 0 else 0)

        # prepare for linear regression
        np_lnt = np.array(lnt).reshape(-1, 1)
        np_lnb = np.array(lnb)

        # linear regression to get power law exponent
        model = LinearRegression()
        model.fit(np_lnt, np_lnb)
        linreg_predict = model.predict(np_lnt)

        value_to_analyze = filename.split('.txt')[0].split('_')[-1]
        
        [directory, filename] = filename.split('/')
        with open(directory + "/hist_" + filename, "w") as f:
            f.write(f"t\tb\tlnt\tlnb\tlinreg\t k=" + str(model.coef_) + ", b=" + str(model.intercept_) + "\n")

            for i in range(len(lnb)):
                f.write(str(bins[i]) + "\t" + str(n[i]) + "\t" + str(lnt[i]) + "\t" + str(lnb[i]) + "\t" + str(linreg_predict[i]) + "\n")
    
        plt.scatter(lnt, lnb)
        plt.title(f'Распределение {value_to_analyze}')
        plt.xlabel('log k')
        plt.ylabel(f'log {value_to_analyze}')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()


if __name__ == "__main__":
    obtain_average_distribution(local_filenames)
