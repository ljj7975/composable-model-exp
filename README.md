# kws-convrnn-exp
exploring variations of convrnn for keyword spotting task

## training model

```
python train.py -bc config/mnist_base.json -fc config/mnist_fine_tune.json -t 1
```

## finetune model

```
python train.py -b saved/mnist_base/0330_155217/model_best.pth -fc config/mnist_fine_tune.json -t 1
```

## evaluation model

```
python evaluate.py -b trained/mnist_base/0330_160337/model_best.pth -t 1 2 -ft trained/mnist_fine_tune
```
