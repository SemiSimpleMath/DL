from matplotlib import pyplot as plt
from transformer_libs import utils
from collections import defaultdict

import config


def get_data(file):
    data = defaultdict(list)
    with open(file) as f:
        for line in f:
            line = line.strip()
            line = line.split(":")
            data[line[0]].append(line[1])
    return data

most_recent_log = utils.most_recent_file(config.log_directory)
d1 = get_data(most_recent_log)
#d1 = get_data('logs/log6783.txt')


# d3 = get_data('logs/log88319.txt')


x1 = d1['batch_num']
y1 = d1['current_loss']

# x3 = d3['batch_num']
# y3 = d3['current_loss']

high = len(x1)
low = high - 500
print (high)
x1 = x1[low:high]
y1 = y1[low:high]

# x3 = x3[:high]
# y3 = y3[:high]

plt.xlabel('Number of batches')
plt.ylabel(f'Average loss')

x1 = list(map(float, x1))
y1 = list(map(float, y1))

from scipy.signal import savgol_filter
yhat = savgol_filter(y1, 10, 3) # window size 51, polynomial order 3
x2 = list(map(float, x1))
y2 = list(map(float, yhat))

# x3 = list(map(float, x3))
# y3 = list(map(float, y3))

plt.plot(x1, y1)
plt.plot(x2, y2)
# plt.plot(x3, y3)
plt.show()
