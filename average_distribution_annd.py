import math
import numpy as np
import matplotlib.pyplot as plt
from average_distribution_window import apply_rolling_window
from collections import defaultdict
from sklearn.linear_model import LinearRegression

local_filenames = [
    #"output/out_ba_25000_3_dist_as.txt", "output/out_ba_25000_3_dist_sig.txt",
    #"output/musae_git_edges_dist_as.txt", "output/musae_git_edges_dist_sig.txt",
    #"output/artist_edges_dist_as.txt", "output/artist_edges_dist_sig.txt",
    #"output/soc-twitter-follows_dist_as.txt", "output/soc-twitter-follows_dist_sig.txt",
    "output/soc-flickr_dist_as.txt", "output/soc-flickr_dist_sig.txt",
    ]


def obtain_average_distributions(filenames, window_size = None):
    for filename in filenames:
        f = open(filename)
        lines = f.readlines()

        deg_val_count = defaultdict(lambda: (0, 0.0))
        for line in lines:
            line = line[1:-2]
            results = line.split(') (')
            for value_degree in results:
                value, degree = value_degree.split(', ')
                value, degree = float(value), round(float(degree), 3)
                deg_val_count[degree] = (deg_val_count[degree][0] + 1, deg_val_count[degree][1] + value)

        deg_val = dict()
        for degree in deg_val_count.keys():
            deg_val[degree] = (deg_val_count[degree][1] / deg_val_count[degree][0])

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

        if window_size and window_size > 1:
            debug_old_values = values
            values = apply_rolling_window(values, window_size)
            debug_old_log_values = log_values
            log_values = apply_rolling_window(log_values, window_size)

        
        # prepare for linear regression
        np_lnk = np.array(log_degrees).reshape(-1, 1)
        np_lnv = np.array(log_values)

        # linear regression to get power law exponent
        model = LinearRegression()
        model.fit(np_lnk, np_lnv)
        linreg_predict = model.predict(np_lnk)

        slope, intercept = round(model.coef_[0], 2), round(model.intercept_, 2)

        [directory, filename] = filename.split('/')
        with open(directory + "/hist_" + filename, "w") as f:
            f.write("k\tv\tlnk\tlnv\tlinreg\t k=" + str(slope) + ", b=" + str(intercept) + "\n")

            for i in range(len(degrees)):
                f.write(str(degrees[i]) + "\t" + str(values[i]) + "\t" + str(log_degrees[i]) + "\t" + str(log_values[i]) + "\t" + str(linreg_predict[i]) + "\n")

        visualize = True

        if visualize:
            metric_type_string = get_metric_type_string(filename)
            set_ylabel_by_metric_type(filename, prefix="log")
            linreg_y = [model.intercept_ + model.coef_ * x for x in log_degrees]
            print("Average distribution LinReg:", log_degrees, linreg_y)
            # plt.scatter(log_degrees, log_values, s=3)
            plt.plot(log_degrees, log_values)
            plt.xlabel("log k")
            plt.title(f"{metric_type_string} в {filename.split('.txt')[0]}")
            plt.plot(log_degrees, linreg_y, "r", label=f'y={slope}x + {intercept}')
            plt.legend()
            plt.show()

            set_ylabel_by_metric_type(filename)
            # plt.scatter(degrees, values, s=3)
            plt.plot(degrees, values)
            plt.xlabel("k")
            plt.title(f"{metric_type_string} в {filename.split('.txt')[0]}")
            plt.legend()
            plt.show()


def get_metric_type_string(filename):
    if "_as" in filename:
        return "Средние"
    elif "_sig" in filename:
        return "Дисперсиии"
    elif "_cv" in filename:
        return "CV"
    
def set_ylabel_by_metric_type(filename, prefix=""):
    if "_as" in filename:
        plt.ylabel(f"{prefix}\Phi")
    elif "_sig" in filename:
        plt.ylabel(f"{prefix}\Theta")
    elif "_cv" in filename:
        plt.ylabel(f"{prefix}CV")


if __name__ == "__main__":
    obtain_average_distributions(local_filenames)
    


