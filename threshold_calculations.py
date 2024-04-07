import math

def calculate_threshold_percentages_by_degree(deg2sum_count, deg2values, filename, 
                                              value_to_analyze_name, threshold = 1):
    degree2result = dict()
    overall_node_count = 0
    overall_node_greater_threshold = 0
    for degree in deg2values.keys():
        # Node count, Value > {threshold}, Value <= {threshold}
        count_greater_threshold = 0
        for value in deg2values[degree]:
            count_greater_threshold += 1 if value > threshold else 0 

        node_count = deg2sum_count[degree][1]
        overall_node_count += node_count
        overall_node_greater_threshold += count_greater_threshold
        degree2result[degree] = (math.log(degree), round(count_greater_threshold / node_count, 4), node_count, count_greater_threshold, node_count - count_greater_threshold)

    overall_node_greater_threshold_share = overall_node_greater_threshold / overall_node_count
    filename_main, ext = filename.split('.txt')
    result_filename = f"{filename_main}_threshold_stats_{value_to_analyze_name}.csv"
    with open(result_filename, "w") as f:
        f.write(f'logdeg\tfiTrue(Overall: {overall_node_greater_threshold_share})\tNode count\tNode share'
                + f'\tValue > {threshold}\tValue <= {threshold}\n')
        degrees_sorted = list(sorted(deg2values.keys()))
        for degree in degrees_sorted:
            result = degree2result[degree]
            node_share_in_graph = result[2] / overall_node_count 
            f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{node_share_in_graph}\t"
                    + f"{result[3]}\t{result[4]}\n")


    
        