## How to

It is assumed that:
- you are using a Unix-like operating system
- Python 3.10 is installed on your machine
- you are connected to internet

### <u>Run the code and reproduce results</u>

At the root of the folder you should be able to reproduce the results by running these make targets:

```bash
make install model1 model2 eval-model1 eval-model2 plot
```

The targets that train the models depends on a target that download the data so you don't have to worry about the dataset.

On a MacBook Pro 2019 (2.6 GHz 6-Core Intel Core i7) it takes approximately:
- 22 minutes to train the deep learning model
- 14 minutes to train the non-deep machine learning model

### <u>Get help</u>

```bash
make help
```