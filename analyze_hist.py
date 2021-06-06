filename = "output/hist_web-google-dir.txt"

### How to run:
# Change filename variable above, insert automatically generated in main.py file that starts with "hist_"
# Return percentage of friendship index values for nodes in the network that are larger than one.

def get_beta_count(datastring):
    return int(datastring.split('\t')[1])

if __name__ == "__main__":
    f = open(filename)
    f.readline() # skip first line
    less_than_one = get_beta_count(f.readline())
    more_than_one = 0
    other_lines = f.readlines()
    for line in other_lines:
        if line:
            more_than_one += get_beta_count(line)
    
    print(f"Percentage of friendship index > 1 is: {more_than_one / (less_than_one + more_than_one)}")