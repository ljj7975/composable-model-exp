# kws-convrnn-exp
exploring variations of convrnn for keyword spotting task

## training model

```
python train.py -bc config/mnist_base.json -fc config/mnist_fine_tune.json
python train.py -bc config/cifar10_base.json -fc config/cifar10_fine_tune.json
```

## finetune model

```
python train.py -b saved/mnist_base/0330_155217/model_best.pth -fc config/mnist_fine_tune.json -t 1
```

## evaluation model

```
python evaluate.py -b saved/mnist_base -ft saved/mnist_fine_tune
```

##  run experiment
```
python experiment.py -t -nm 10
```
