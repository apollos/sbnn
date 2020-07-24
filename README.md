# Stochastic Bayesian Neural Networks
Code for the paper `Stochastic Bayesian Neural Networks`.

Link to [Paper](https://abhinavsagar.github.io/files/sbnn.pdf).

## Data

The dataset can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets.php?format=&task=reg&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table).

## Dependencies

1. Tensorflow
2. GPflow-Slim

## Algorithm

![roc-auc](images/img1.png)

## Installation

`python exp/regression.py -d yacht`

## Results
 
Averaged test RMSE for the regression benchmarks
 
![roc-auc](images/img2.png)

Averaged log-likelihood for the regression benchmarks

![roc-auc](images/img3.png)

## Citing

If you find this code useful in your research, please consider citing the paper:

```
@article{sagarstochastic,
  title={Stochastic Bayesian Neural Networks},
  author={Sagar, Abhinav}
}
```

## License

```
MIT License

Copyright (c) 2020 Abhinav Sagar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
