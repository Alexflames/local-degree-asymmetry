from dataclasses import dataclass

filename = "ia-facebook-wall-wosn-dir.edges"
delimiter = ' '
from_id = 0
to_id = 1
timestamp_id = 3

@dataclass
class Edge:
    node_from: int
    node_to: int
    timestamp: int

    def __lt__(self, other):
        return self.timestamp < other.timestamp


def read_file(filename):
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("%"):
                continue
            line_split = line.split(delimiter)
            node_from = int(line_split[from_id])
            node_to = int(line_split[to_id])
            timestamp = int(line_split[timestamp_id])
            edges.append(Edge(node_from, node_to, timestamp))
    return edges


def sort_data(edges_unsorted):
    return sorted(edges_unsorted)


def renumber_ascending(edges_sorted):
    mapping = dict()
    edges_renumbered = list()
    
    i = 1
    for edge in edges_sorted:
        value_in_dict_from = mapping.get(edge.node_from)
        # Присваиваем новый номер началу ребра при необхдимости
        if value_in_dict_from == None:
            mapping[edge.node_from] = i
            i += 1
        # Присваиваем новый номер концу ребра при необхдимости
        value_in_dict_to = mapping.get(edge.node_to)
        if value_in_dict_to == None:
            mapping[edge.node_to] = i
            i += 1

        # Перенумеровываем и записываем в массив ребер
        edges_renumbered.append(
            Edge(mapping[edge.node_from], mapping[edge.node_to], edge.timestamp))
    return edges_renumbered


def write_to_file(edges_sorted, filename_original):
    original_name, original_ext = filename_original.split('.')
    with open(original_name + "-sorted." + original_ext, 'w') as f:
        for edge in edges_sorted:
            f.write(f"{edge.node_from} {edge.node_to} {edge.timestamp}\n")


if __name__ == "__main__":
    edges_unsorted = read_file(filename)
    
    edges_sorted = sort_data(edges_unsorted)
    
    edges_renumbered = renumber_ascending(edges_sorted)
    
    write_to_file(edges_renumbered, filename)