# Dynamic Memory Networks in Tensorflow
Implementation of [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://arxiv.org/abs/1506.07285)
on the [bAbI tasks](https://research.facebook.com/research/babi/) using Tensorflow.

![DMN Structure](http://i.imgur.com/Gt73C4X.png)

In the paper, DMN does bAbi task well with strongly supervised settings. But in this implementation,
I've chosen weakly supervised setting - training DMN without supporting facts information in bAbI.

## Prerequisites
- Python 3.x
- Tensorflow 0.8+
- Numpy
- [tqdm](https://pypi.python.org/pypi/tqdm) - Progress bar module

## Usage
First, Install it.
```
sudo pip install tqdm
git clone https://github.com/therne/dmn-tensorflow
```

### Training the model
```
./main.py --task [bAbi Task Number]
```

### Test the model
```
./main.py --test --save_dir [Saved directory]
```

### Results

*Model trained with 40 hidden units, semantic memory, 5 memory step, 300 epoches and learning rate 2e-2.*

Task    | Result
--------|-------
1. Single supporting facts | 100%
2. Two supporting facts    | 92.44%
3. Three supporting facts  | 92.44%
4. Two arguments relations | 98.89%
5. Three arguments relations | 98.79%
6. Yes-No Questions        | 99.09%
7. Counting                | 98.39%
8. List/Sets               | 99.19%
9. Simple negotiation      | 98.49%
10. Indefinite knowledge   | 94.86%
11. Basic coreference	     | 99.50%
12. Conjuction             | 98.69%
13. Compound coreference   | 99.09%
14. Time reasoning         | 98.19%
15. Basic deduction        | 93.75%
16. Basic induction        | 94.05%
17. Positional reasoning   | 96.17%
18. Size reasoning         | 100%
19. Path finding           | 77.72%
20. Agent’s motivations    | 100%
Average                    | 96.48%

Result comes close to the fully-supervised settings. 

### References
- [Implementing Dynamic memory networks by YerevaNN][impl-dmn-yerevann] - Original work, great article that helped me a lot
- [Dynamic-memory-networks-in-Theano][dmn-in-theano] - Original work.
- [memnn-tensorflow][memnn-tensorflow] - Base code (trainer, loader)

### TO-DO
- Use [`tf.nn.dynamic_rnn`][dynamic-rnn-docs] for memory optimization
- Use GRU in answer module to generate answer sequence
- Interactive mode?

[impl-dmn-yerevann]: https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/
[dmn-in-theano]: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
[memnn-tensorflow]: https://github.com/seominjoon/memnn-tensorflow
[dynamic-rnn-docs]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.dynamic_rnn.md
