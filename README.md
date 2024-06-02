# DCMSL: Dual influenced Community Strength-boosted Multi-Scale Graph Contrastive Learning

PyTorch implementation for KBS Under Review paper "DCMSL: Dual influenced Community Strength-boosted Multi-Scale Graph Contrastive Learning".

# Requirements
* Python 3.8.8
* PyTorch 1.8.1
* torch_geometric 2.0.1
* cdlib 0.2.6
* networkx 2.5.1
* numpy 1.22.4

# Running the code
The hyperparameters for node classification can be found in `./param`, which will be directly loaded by `--param`:

~~~
python train.py --dataset Coauthor-CS --param local:coauthor_cs.json 
~~~

It can be changed the parameter by either editting .json files or adding it to the command, for example:

```shell
python train.py --dataset Coauthor-CS --param local:coauthor_cs.json  --delta 0.3
```
