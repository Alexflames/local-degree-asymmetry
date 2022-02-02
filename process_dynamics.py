import matplotlib.pyplot as plt

filenames_internal = [#"output/out_ba_25000_3_10_a.txt", "output/out_ba_25000_3_50_a.txt", "output/out_ba_25000_3_100_a.txt", "output/out_ba_25000_3_1000_a.txt",
            "output/out_ba_25000_5_10_a.txt", "output/out_ba_25000_5_50_a.txt", "output/out_ba_25000_5_100_a.txt", "output/out_ba_25000_5_1000_a.txt"]

# this code averages result for nodes and produces proc_ file that contains LaTeX Tikzpicture-compatible data 
# for visualising friendship index dynamics

def process_dynamics(filenames):
    x_ranges = []
    trajectories = []
    for filename in filenames:
        start_from = filename.split('.txt')[0].split('_')[-2]
        x_range = range(int(start_from), int(filename.split('_')[2]), 50)
    
        metric_type = filename.split('.txt')[0].split('_')[-1]

        f = open(filename)
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

        f_out = open("output/" + "proc_" + filename.split('/')[1], "w")
        f_out.write("t\t" + metric_type + "(t)\n")
        for i in range(len(processed_values)):
            f_out.write(str(x_range[i]) + "\t" + str(processed_values[i]) + "\n")

        f_out.close()
        f.close()
        
        x_ranges.append(x_range)
        trajectories.append(processed_values)


    metric_type = filenames[0].split('.txt')[0].split('_')[-1]
    node_numbers = [filename.split('.txt')[0].split('_')[-2] for filename in filenames]
    plt.title(f"Траектории {metric_type} для узлов {', '.join(node_numbers)}")
    plt.xlabel("t")
    plt.ylabel(metric_type)
    if len(filenames) == 1:
        plt.plot(x_ranges[0], trajectories[0], 'r')
    if len(filenames) == 2:
        plt.plot(x_ranges[0], trajectories[0], 'r', x_ranges[1], trajectories[1], 'g')
    if len(filenames) == 3:
        plt.plot(x_ranges[0], trajectories[0], 'r', x_ranges[1], trajectories[1], 'g', x_ranges[2], trajectories[2], 'b')
    if len(filenames) == 4:
        plt.plot(x_ranges[0], trajectories[0], 'r', x_ranges[1], trajectories[1], 'g', x_ranges[2], trajectories[2], 'b', x_ranges[3], trajectories[3], 'm')
    plt.show()


def process_s_a_b_dynamics(filenames):
    if len(filenames) == 0:
        return

    for i in range(len(filenames[0])):
        files = [filepack[i] for filepack in filenames]
        filenames_to_process = [file.name for file in files]
        for file in files:
            file.close() 
        process_dynamics(filenames_to_process)



if __name__ == "__main__":
    process_dynamics(filenames_internal)