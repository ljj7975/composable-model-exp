import json
import pprint
from matplotlib import pyplot as plt
import numpy as np

summary_file = "summary.txt"

results = eval(open(summary_file).read())

EXP_LOSS = ['logsoftmax_nll_loss', 'softmax_bce_loss', 'sigmoid_bce_loss']
num_model = results['num_model']

print("num_model", num_model)

fig = plt.figure(figsize=(7,4))

legends = []

for loss in EXP_LOSS:
    value = results[loss]

    values = value['combined_model']['average_accuracy']
    num_class = results.get('num_class', len(values))
    legend = plt.plot(np.arange(num_class) + 1, values)
    legends.append(legend)

font_size = 13
plt.xlabel('number of classes', fontsize=font_size)
plt.ylabel('accuracy (%)', fontsize=font_size)

plt.ylim(75, 100)

fig.legend(legends,     # The line objects
           labels=EXP_LOSS,
           loc=8,  # Position of legend
           ncol=3
           )
plt.grid()
fig.subplots_adjust(bottom=0.22, top=0.95)
fig.savefig("mnist.png")
plt.show()