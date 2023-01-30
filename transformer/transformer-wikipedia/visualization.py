from matplotlib import pyplot as plt

from collections import defaultdict


def get_data(file):
    data = defaultdict(list)
    with open(file) as f:
        for line in f:
            line = line.strip()
            line = line.split(":")
            data[line[0]].append(line[1])
    return data


d2 = get_data('logs/log91961.txt')
d1 = get_data('logs/log55143.txt')

x1 = d1['batch_num']
y1 = d1['current_loss']

x2 = d2['batch_num']
y2 = d2['current_loss']

high = len(x1)

x1 = x1[:high]
y1 = y1[:high]

x2 = x2[:high]
y2 = y2[:high]

plt.xlabel('Number of batches')
plt.ylabel('Average loss over last 200 batches')

x1 = list(map(float, x1))
y1 = list(map(float, y1))

x2 = list(map(float, x2))
y2 = list(map(float, y2))

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.show()
