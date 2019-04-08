import json
import pprint
from matplotlib import pyplot as plt
import numpy as np

task = "kws"
summary_file = task+".txt"

results = eval(open(summary_file).read())

EXP_LOSS = ['logsoftmax_nll_loss', 'softmax_bce_loss', 'sigmoid_bce_loss']
num_model = results['num_model']

print("num_model", num_model)

fig = plt.figure(figsize=(7,4))

legends = []

for loss in EXP_LOSS:
    value = results[loss]
    print(loss)
    print("   base model")
    base_acc = value['base_model']['average_accuracy']
    print("\t", "avg:", base_acc)

    print("   fine_tuned_model")
    ft_values = value['fine_tuned_model']['average_accuracy']
    print("\t", "avg:", np.mean(ft_values))
    print("\t", "min:", np.min(ft_values))
    print("\t", "max:", np.max(ft_values))
	
    values = value['combined_model']['average_accuracy']
    print("   combined model")
    print("\t", "final acc:", values[-1])
    gap = base_acc - values[-1]
    print("\t", "gap from base:", round(100 * gap / base_acc, 2), "%")

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
fig.savefig(task+".png")
plt.show()
