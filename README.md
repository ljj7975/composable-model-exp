# kws-convrnn-exp
exploring variations of convrnn for keyword spotting task

## training model

```
python train.py -bc config/mnist_base.json -fc config/mnist_fine_tune.json
```

## evaluation model

```
python evaluate.py -b saved/mnist_base/0326_162639/model_best.pth -t 1 2
```