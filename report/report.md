## Model 1 (deep learning model)
### Data Preprocessing

### Selected features

### Model choice
sdfsdfs
accuracy
## Model 2 (non-deep machine learning model)
### Data Preprocessing
The engineered features being already normalized and bounded within [-1, 1], I haven't performed any additional preprocessing. I did notice that some columns in the training set do not reach the boundaries, which makes me think that it has been preprocessed taking into account the maximum/minimum observed across the training and the test set. I'll look into this after I've submitted this assignment because if that's indeed the case this would be unfortunate given that this is a form of data leakage. [TODO]

### Selected features
I eventually decided not to take into account at all the features containing the keywords: `meanFreq`, `maxInds` & `bandsEnergy`. I acknowledge this is a large number of features (144) but the reality is that I didn't have enough time to look into what they mean. For the other features, this is what I've done:

[TODO physics] Because it was difficult to hypothesize which features were more likely to have predictive power, my approach has been to group them when I thought they were describing similar concepts, then to perform a form a statistical feature selection among each group.

Features which aren't an axis projection have been grouped together to form the base of the feature space (cf `features.base` in __model2.toml__).

The others have been grouped depending on how they intend to characterize the signal: location (central or non-central), dispersion, auto-correlation or shape.

Each of the 5 time domain variables and 3 frequency domain variables has multiple corresponding in __model2.toml__ (eg: `features.time_domain_groups.tBodyAcc_location`, `features.time_domain_groups.tBodyAcc__dispersion` etc) corresponding to these sub-groups.

Only 2 feature per group are eventually kept by the model. This is controlled by the parameter `Classifier.max_extra_feat` in __model2.toml__.

### Model choice

I haven't formally verified the presence of colinearity in this dataset, but given the large number of variables, it is likely. Because random forests are suitable in presence of correlated features, I've opted for this algorithm. Another advantage is that they are parallelisable, which is helpful to speed up hyperparameter optimisation, such as finetuning the number of trees (cf `training.hyperparameter_grid` in __model2.toml__).

## Pros and cons

|             | Deep learning model | Random Forest |
| ----------- | ------------------- | ------------- |
| Pros        | No feature engineering<br>Higher accuracy | Faster training<br>Better interpretability<br>Easier set up |
| Cons        | Slower training<br>Higher accuracy<br>Need validation set | Heavy feature engineering |

## Results

```make eval-model1```

```
Model accuracy 93.15%

                      1 WALKING  2 WALKING_UPSTAIRS  3 WALKING_DOWNSTAIRS  4 SITTING  5 STANDING  6 LAYING
1 WALKING                   463                   3                    29          1           0         0
2 WALKING_UPSTAIRS            0                 444                    27          0           0         0
3 WALKING_DOWNSTAIRS          4                   3                   413          0           0         0
4 SITTING                     0                   2                     0        400          89         0
5 STANDING                    0                   0                     0         19         513         0
6 LAYING                      0                   0                     0          0          25       512
```

```make eval-model2```

```
Model accuracy 91.45%

                      1 WALKING  2 WALKING_UPSTAIRS  3 WALKING_DOWNSTAIRS  4 SITTING  5 STANDING  6 LAYING
1 WALKING                   485                   3                     8          0           0         0
2 WALKING_UPSTAIRS           65                 398                     8          0           0         0
3 WALKING_DOWNSTAIRS         25                  58                   337          0           0         0
4 SITTING                     0                   0                     0        426          65         0
5 STANDING                    0                   0                     0         20         512         0
6 LAYING                      0                   0                     0          0           0       537
```

(as you would communicate to a non-technical colleague. Include 1
visualisation of your choice to illustrate the results.)

![sitting/sitting vs sitting/standing](parallel_coordinates.png)

