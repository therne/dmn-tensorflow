# Dynamic Memory Networks in Tensorflow
Implementation of [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://arxiv.org/abs/1506.07285)
on the [bAbI tasks](https://research.facebook.com/research/babi/) using Tensorflow.

![DMN Structure](http://i.imgur.com/Gt73C4X.png)

In the paper, DMN does bAbi task well with strongly supervised settings. But in this implementation,
I've chosen weakly supervised setting - training DMN without supporting facts information in bAbI.

## Prerequisites
- Tensorflow
- Numpy
- [tqdm](https://pypi.python.org/pypi/tqdm)

### Training the model
```
./main.py --task [bAbi Task Number]
```

### Test the model
```
./main.py --test --save_dir [Saved directory]
```

### Results
TODO: Add test result

### References
- [Implementing Dynamic memory networks by YerevaNN](https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/) - Original, and great article
