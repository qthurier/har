## Model 1 (deep learning model)
### Data Preprocessing

### Selected features

### Model choice
sdfsdfs

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
(as you would communicate to a non-technical colleague. Include 1
visualisation of your choice to illustrate the results.)

