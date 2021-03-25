import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression

filenames = [
    #"output/out_ba_25000_3_dist_a.txt", "output/out_ba_25000_3_dist_sig.txt",
    #"output/musae_git_edges_dist_a.txt", "output/musae_git_edges_dist_sig.txt",
    #"output/artist_edges_dist_a.txt", "output/artist_edges_dist_sig.txt"
    #"output/soc-twitter-follows_dist_a.txt", "output/soc-twitter-follows_dist_sig.txt"
    "output/soc-flickr_dist_a.txt", "output/soc-flickr_dist_sig.txt"
    ]

if __name__ == "__main__":
    for filename in filenames:
        f = open(filename)
        lines = f.readlines()

        deg_val_count = defaultdict(lambda: (0, 0.0))
        for line in lines:
            line = line[1:-2]
            results = line.split(') (')
            for value_degree in results:
                value, degree = value_degree.split(', ')
                value, degree = float(value), int(degree)
                deg_val_count[degree] = (deg_val_count[degree][0] + 1, deg_val_count[degree][1] + value)

        deg_val = dict()
        for degree in deg_val_count.keys():
            deg_val[degree] = (deg_val_count[degree][1] / deg_val_count[degree][0])

        #print(deg_val_count)
        #print(deg_val)
        degrees = list(deg_val.keys())
        values = list(deg_val.values())
        # filter zeroes and sort
        deg_vals = sorted(filter(lambda y: y[1] > 0, zip(degrees, values)), key=lambda x: x[0])

        degrees, values = [], []        
        for i in range(len(deg_vals)):
            degrees.append(deg_vals[i][0])
            values.append(deg_vals[i][1])

        log_degrees, log_values = [], []
        for i in range(len(degrees)):
            log_degrees.append(math.log10(degrees[i]))
            log_values.append(math.log10(values[i]))
        
        # prepare for linear regression
        np_lnk = np.array(log_degrees).reshape(-1, 1)
        np_lnv = np.array(log_values)

        # linear regression to get power law exponent
        model = LinearRegression()
        model.fit(np_lnk, np_lnv)
        linreg_predict = model.predict(np_lnk)

        metric_type = ""
        if "_a" in filename:
            plt.ylabel("alpha")
            metric_type = "a"
        elif "_sig" in filename:
            plt.ylabel("sigma")
            metric_type = "sig"

        # f_out = open("output/proc_" + filename.split('/')[:-1], "w")
        # f_out.write("k\t" + metric_type + "\n")
        # for i in range(len(degrees)):
        #     f_out.write(str(degrees[i]) + "\t" + str(degrees[i]) + "\n")

        [directory, filename] = filename.split('/')
        with open(directory + "/hist_" + filename, "w") as f:
            f.write("k\tv\tlnk\tlnv\tlinreg\t k=" + str(model.coef_) + ", b=" + str(model.intercept_) + "\n")

            for i in range(len(degrees)):
                f.write(str(degrees[i]) + "\t" + str(values[i]) + "\t" + str(log_degrees[i]) + "\t" + str(log_values[i]) + "\t" + str(linreg_predict[i]) + "\n")

        # plt.scatter(degrees, values, s=3)
        # plt.xlabel("degree")
        # plt.title("Averaged distribution of values on Barabasi-Albert")
        # ax = plt.gca()
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # ax.set_ylim([0.1, 10000])
        # plt.show()