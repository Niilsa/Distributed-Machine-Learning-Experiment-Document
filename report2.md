# Experiment 2 (task 3 + 4)

## Data Parallelism 

### Experiment content

* Understand common data partitioning strategies;

* Write corresponding algorithm implementation code for data partitioning;

* Analyze the impact of different data partitioning methods on the model.

### Experimental requirements

1. Simulate a multi node DML system, partition a complete dataset in different ways, and allocate it to the nodes in the system.

2. Implement partitioning methods that include random sampling and random partitioning, and train models in parallel with data

3. Analyze the performance improvement of data parallelism compared to single machine training, and analyze the impact of different data partitioning methods on model performance

### Overview

For this task I have implemented and compare two samplers: simple `RandomSampler` which shuffle all data samples and `BalancedSampler` which ensures each process gets an approximately equal number of unique samples each epoch.

The `BalancedSampler` class is similar to the `RandomSampler` class, but with a difference: The `BalancedSampler` selects a balanced subset of the dataset's indices for each replica to ensure that each replica gets an approximately equal number of unique samples each epoch. This is done in the `__iter__` method by using slicing to select a unique subset of indices for each replica.

### Results

`RandomSampler`:

![RandomSampler](/home/niilsa/Documents/github/Distributed-Machine-Learning-Experiment-Document/report_images/RandomSampler.png)

`BalancedSampler`:

![BalancedSampler](/home/niilsa/Documents/github/Distributed-Machine-Learning-Experiment-Document/report_images/BalancedSampler.png)

### Analysis

#### Accuracy and Loss Value

Both `RandomSampler` and `BalancedSampler` have resulted in approximately the same accuracy and loss values. This indicates that from a pure model performance perspective, the difference in the sampling strategies has not significantly impacted the final model's performance. Both strategies ensure all data points are being considered and none are being overlooked.

#### Data Distribution

If the data distribution is skewed or imbalanced, `BalancedSampler` could potentially provide a more equitable distribution of data points  across processes. This might not impact the accuracy or loss but can  ensure that each process gets a fair representation of the data.

#### Conclusion

In conclusion, while both samplers have resulted in similar accuracy and loss values, the final choice between them would depend on these other factors. It is essential to balance performance, efficiency, scalability, and robustness when choosing the best sampling strategy for a specific application.



## Model Parallelism

### Experimental content

* Understand common model partitioning strategies;

* Write corresponding algorithm implementation code for model partitioning;

* Analyze the impact of different data partitioning methods on the model.

### Experimental requirements

1. Using RPC related APIs to achieve parallel model training

2. Split the model into two parts and train them separately on different nodes (processes)

3. Analyze the impact of model parallelism on distributed system performance based on experimental results

### Result

![task4_start](/home/niilsa/Documents/github/Distributed-Machine-Learning-Experiment-Document/report_images/task4_start.png)

![task4_end](/home/niilsa/Documents/github/Distributed-Machine-Learning-Experiment-Document/report_images/task4_end.png)

### Analysis

**Model Splitting**: We split our Convolutional Neural Network model into two parts: convolutional layers and fully connected layers. The convolutional layers were hosted on Node 1 and the fully connected layers on Node 2.

**RPC Setup**: The RPC framework was set up successfully, allowing Node 1 and Node 2 to communicate with each other for both the forward and backward passes during training.

**Parallel Training**: The model was trained successfully in parallel on the two nodes using the PyTorch RPC framework.

**Accuracy**: Model accuracy for "model parallel" approach is similar to other methods, while time is  higher than for "allgather" approach. 