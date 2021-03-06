# Composing Algorithm
In this work, I introduce **Composing algorithm** enabling dynamic construction of a neural network classifier using class-level transfer. Details can be found in [the write up](https://github.com/ljj7975/composable-model-exp/blob/master/report/project.pdf)

Composing algorithm has following steps:

1. Pre-train a model using every class
2. Freeze the model parameters
2. Construct a new dataset which labels one class as positive and the others as negative
3. Replace the last full-connected layer for two classes
4. Fine-tune the last layer using the new dataset
5. Retrieve the weights for positive class from the last fully-connected layer
6. Repeat step 3 ~ 6 for each class
7. For every combination of target classes, it is possible to obtain a classifier by reconstructing the last layer with class-specific weights obtained from fine-tuning

![Alt text](https://github.com/ljj7975/composable-model-exp/blob/master/report/composing_algo.png)

## Experiments

I also explore the feasibility of Composing algorithm on MNIST, Keyword Spotting and CIFAR-100.

- Loss function plays a key role in Composing algorithm and sigmoid with BCE loss minimizes accuracy degradation between multi-class model and composed model.

- Decrease in accuracy is inevitable as number of classes increases.

### training base model and fine-tune for each class
```
python train.py -bc <base model config> -fc <fine tuning model config>
```

### fine-tune model
```
python train.py -b <path to base model> -fc <fine tuning model config> -t <targets>
```

### evaluation model
```
python evaluate.py -b <path to base model> -ft <path to fine-tuned model>
```

### train and run experiments
```
python experiment.py -t -nm <number of models to train> -ni <number of iterations for experiments> -e <experiments to run>
```
