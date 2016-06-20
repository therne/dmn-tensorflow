# Dynamic Memory Networks in Tensorflow
![DMN Structure][img-url]

Implementation of [Dynamic Memory Networks for Visual and Textual Question Answering][paper] on the
[bAbI question answering tasks][babi] using Tensorflow.

## Prerequisites
- Python 3.x
- Tensorflow 0.8+
- Numpy
- [tqdm](https://pypi.python.org/pypi/tqdm) - Progress bar module

## Usage
First, You need to install dependencies.
```
sudo pip install tqdm
git clone https://github.com/therne/dmn-tensorflow & cd dmn-tensorflow
```

Then download the dataset:
```
mkdir data
curl -O http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -xzf tasks_1-20_v1-2.tar.gz -C data/
```

If you want to run original DMN (`models/old/dmn.py`), you also need to download GloVe word embedding data. you can download it by running `fetch_glove.sh`.

### Training the model
```
./main.py --task [bAbi Task Number]
```

### Testing the model
```
./main.py --test --task [Task Number]
```

### Results
*Single run of DMN+ model trained with paper settings (Batch 128, 3 episodes, 80 hidden, dropout rate 0.9, L2) + batch normalization. The skipped tasks achieved 0 error.*

Task                         | Error Rate
-----------------------------|-------
2. Two supporting facts      | 27.2%
3. Three supporting facts    | -
4. Two arguments relations   | 23.4%
5. Three arguments relations | -
8. List/Sets                 | 0.4%
13. Compound coreference     | 1.5%
14. Time reasoning           | 0.8%
16. Basic induction          | 65.9%
17. Positional reasoning     | 19.2%
18. Size reasoning           | 8.7%
19. Path finding             | 69.9%
Average                      | 10.9%

Overfitting occurs in some tasks and error rate is higher than the paper's result.
I think we need some additional regularizations.

### References
- [Implementing Dynamic memory networks by YerevaNN][impl-dmn-yerevann] - Great article that helped me a lot
- [Dynamic-memory-networks-in-Theano][dmn-in-theano]

### To-do
- More regularizations and hyperparameter tuning
- Visual question answering
- Attention visualization
- Interactive mode?

[paper]: https://arxiv.org/abs/1603.01417
[babi]: https://research.facebook.com/research/babi/
[img-url]: http://i.imgur.com/30DePKh.png
[impl-dmn-yerevann]: https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/
[dmn-in-theano]: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
