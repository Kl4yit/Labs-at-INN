"""
 1. Make dataGenerator
    a. pass the data Amount
    b. pass the noise level
 2. Create model
 3. Train model
 4. Test model
"""
"""
 Calling fit on the pipeline is the same as calling fit on each estimator in turn, 
 transform the input and pass it on to the next step. The pipeline has all the methods 
 that the last estimator in the pipeline has, i.e. if the last estimator is a classifier, the 
 Pipeline can be used as a classifier. If the last estimator is a transformer, again, so is 
 the pipeline.
"""
import numpy as np
from datagenerator import DataGenerator
generator = DataGenerator(noise_level=3)

(train_data, train_labels) = generator.get_train_data(data_amount=300)
(test_data, test_labels) = generator.get_test_data(data_amount=100)

(n_samples, n_features), n_digits = train_data.shape,  np.unique(train_labels).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")