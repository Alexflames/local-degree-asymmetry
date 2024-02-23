import math

def calculate_friendship_paradox_percentages_by_degree(deg2sum_count, deg2betas, filename):
    degree2result = dict()
    for degree in deg2betas.keys():
        # Node count, Beta > 1, Beta <= 1
        beta_greater1 = 0
        for beta in deg2betas[degree]:
            beta_greater1 += 1 if beta > 1 else 0 

        node_count = deg2sum_count[degree][1]
        degree2result[degree] = (math.log(degree), beta_greater1 / node_count, node_count, beta_greater1, node_count - beta_greater1)

    filename_main, ext = filename.split('.')
    result_filename = f"{filename_main}_beta_stats.csv"
    with open(result_filename, "w") as f:
        f.write('logdeg\tfiTrue\tNode count\tBeta > 1\tBeta <= 1\n')
        degrees_sorted = list(sorted(deg2betas.keys()))
        for degree in degrees_sorted:
            result = degree2result[degree]
            f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{result[4]}\n")


    
        