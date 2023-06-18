# Experiment 3 (task 2)

## Communication Model and Parameter Aggregation

### Experiment content

* Understand common messaging interfaces in collective communication (Reduce, AllReduce, Gather, AllGather, Scatter et al.);

* Based on collective communication, model parameters (gradients) are aggregated and updated, attempting to use different aggregation methods (Sum, Mean, etc.) to aggregate model parameters and gradients;

* Analyze the impact of different aggregation strategies on model performance.


### Experimental requirements

1. Implement parameter (gradient) aggregation under collective communication, implement gradient averaging aggregation methods based on at least 2 collective communication primitives, compare their communication time costs, and analyze the impact of different aggregation strategies on model performance

2. Set bottleneck nodes under the framework and discuss the impact of bottleneck nodes on collective communication.

### Results

**allreduce_average_gradients** method:

![allreduce](/home/niilsa/Documents/github/Distributed-Machine-Learning-Experiment-Document/report_images/allreduce.png)

**allgather_average_gradients** method:

![allgather](/home/niilsa/Documents/github/Distributed-Machine-Learning-Experiment-Document/report_images/allgather.png)

### Analysis

Both methods produce similar results in term of accuracy. Allreduce method works faster than Allgather. 

Allreduce is generally considered more efficient because it combines the processes of reducing data (such as sum or average) and broadcasting the result to all participating nodes simultaneously. Allgather, on the other hand, involves gathering data from each node and distributing the combined data back to all nodes. 

Thus, it is generally expected that Allreduce would be faster due to its simultaneous reduction and broadcast nature.