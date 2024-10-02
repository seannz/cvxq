import math
table = ""
for bit in [3, 4]:
    for group_size in reversed([1024, 512, 256, 128, 64]):
        for embed in [1024, 2048, 5120, 7168]:
            groups = embed // group_size
            overhead = ((groups * (2 * 16 + 3) + math.ceil(math.log2(groups))) * (4 * embed + embed + 4 * embed)) / (bit * (4 * embed * embed + embed * 4 * embed + 4 * embed * embed)) * 100
            table += ("%5.2f" % overhead)
            table += "\t"
        table += "\n"
    table += "\n"

print(table)
