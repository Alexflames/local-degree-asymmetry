import matplotlib.pyplot as plt

filenames = [#"out_ba_25000_3_10_a.txt", "out_ba_25000_3_50_a.txt", "out_ba_25000_3_100_a.txt", "out_ba_25000_3_1000_a.txt",
            "out_ba_25000_5_10_a.txt", "out_ba_25000_5_50_a.txt", "out_ba_25000_5_100_a.txt", "out_ba_25000_5_1000_a.txt"]

# this code averages result for nodes and produces proc_ file that contains LaTeX Tikzpicture-compatible data 
# for visualising friendship index dynamics

if __name__ == "__main__":
    x_ranges = []
    trajectories = []
    for filename in filenames:
        start_from = filename.split('.')[0].split('_')[-2]
        x_range = range(int(start_from), int(filename.split('_')[2]), 50)
        metric_type = filename.split('.')[-2].split('_')[-1]

        f = open("output/" + filename)
        lines = f.readlines()
        # Temporary
        processed_values = [0 for x in lines[1].split(' ')]
        data_count = 0
        for line in lines:
            if line.strip() and not (line.startswith(">")):
                data_count += 1
                values = line.split(' ')
                for i in range(len(values)):
                    processed_values[i] += float(values[i])
        
        for i in range(len(processed_values)):
            processed_values[i] /= data_count

        f_out = open("output/" + "proc_" + filename, "w")
        f_out.write("t\t" + metric_type + "(t)\n")
        for i in range(len(x_range)):
            f_out.write(str(x_range[i]) + "\t" + str(processed_values[i]) + "\n")

        x_ranges.append(x_range)
        trajectories.append(processed_values)
        f_out.close()
        f.close()

    plt.plot(x_ranges[0], trajectories[0], 'b', x_ranges[1], trajectories[1], 'g', x_ranges[2], trajectories[2], 'r', x_ranges[3], trajectories[3], 'm')
    plt.show()
