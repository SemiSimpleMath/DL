from matplotlib import pyplot as plt
from transformer_libs import utils
from collections import defaultdict
from scipy.signal import savgol_filter
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

x1 = d1['batch_num']
y1 = d1['current_loss']

high = len(x1)
low = high - 100
print(high)
x1 = x1[low:high]
y1 = y1[low:high]

plt.xlabel('Number of batches')
plt.ylabel(f'Average loss')

x1 = list(map(float, x1))
y1 = list(map(float, y1))

yhat = savgol_filter(y1, 10, 3)  # window size 51, polynomial order 3
x2 = list(map(float, x1))
y2 = list(map(float, yhat))

plt.plot(x1, y1)
plt.plot(x2, y2)

plt.show()
