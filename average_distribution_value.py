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
        
        dist = [0 for x in range(50000)]
        for line in lines:
            data = line.split(' ')
            if '|' not in line:
                raise Exception("Invalid string format. Format should be count|value. Maybe you are using old data file and it needs to be cleaned")
            
            for i in range(len(data)):
                n, bin = [round(float(x)) for x in data[i].split('|')]
                if i == 0:
                    less_one += n
                else:
                    over_one += n
                dist[bin] += n
        print(f"Values over one: {round((over_one / (less_one + over_one) * 100), 3)}%")

        lines_count = len(lines)
        dist = [v / lines_count for v in dist]

        # leave only non-zero
        n_bins = zip(dist, range(len(dist)))
        n_bins = list(filter(lambda x: x[0] > 0.01, n_bins))
        n, bins = [ a for (a,b) in n_bins ], [ b for (a,b) in n_bins ]
        
        # get log-log scale distribution
        lnt, lnb = [], []
        for i in range(len(bins)):
            if (n[i] != 0):
                lnt.append(math.log10(bins[i]) if bins[i] != 0 else 0)
                lnb.append(math.log10(n[i]) if n[i] != 0 else 0)

        # prepare for linear regression
        np_lnt = np.array(lnt).reshape(-1, 1)
        np_lnb = np.array(lnb)

        # linear regression to get power law exponent
        model = LinearRegression()
        model.fit(np_lnt, np_lnb)
        linreg_predict = model.predict(np_lnt)
        print(f"Linreg: Coef: {model.coef_}, Intercept: {model.intercept_}")

        value_to_analyze = filename.split('.txt')[0].split('_')[-1]
        
        [directory, filename] = filename.split('/')
        with open(directory + "/hist_" + filename, "w") as f:
            f.write(f"t\tb\tlnt\tlnb\tlinreg\t k=" + str(model.coef_) + ", b=" + str(model.intercept_) + "\n")

            for i in range(len(lnb)):
                f.write(str(bins[i]) + "\t" + str(n[i]) + "\t" + str(lnt[i]) + "\t" + str(lnb[i]) + "\t" + str(linreg_predict[i]) + "\n")
    
        plt.scatter(lnt, lnb)
        plt.plot(lnt, linreg_predict)
        plt.title(f'Распределение {value_to_analyze}')
        plt.xlabel('log k')
        plt.ylabel(f'log {value_to_analyze}')
        ax = plt.gca()
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        plt.show()


if __name__ == "__main__":
    obtain_average_distribution(local_filenames)
