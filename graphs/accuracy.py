import json
import pprint
from matplotlib import pyplot as plt
import numpy as np

summary_file = "summary.txt"

results = eval(open(summary_file).read())

EXP_LOSS = ['logsoftmax_nll_loss', 'softmax_bce_loss', 'sigmoid_bce_loss']
num_model = results['num_model']

print("num_model", num_model)

fig, ax = plt.subplots()

legends = []

for loss in EXP_LOSS:
    value = results[loss]

    values = value['combined_model']['average_accuracy']
    num_class = results.get('num_class', len(values))
    legend = ax.plot(np.arange(num_class) + 1, values)
    legends.append(legend)


ax.set(xlabel='number of classes', ylabel='accuracy (%)', title='Change in accuracy w.r.t change in number of classes')
ax.set_ylim(75, 100)

fig.legend(legends,     # The line objects
           labels=EXP_LOSS,
           loc=8,  # Position of legend
           ncol=3
           )

ax.grid()
fig.subplots_adjust(bottom=0.18, top=0.9)
fig.savefig("mnist.png")
plt.show()