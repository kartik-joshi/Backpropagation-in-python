## Backpropagation implementation with cost matrix ##
# Asymmetric and symmetric cost matrix #
Dermatology dataset is used to train a backprop network here. Dermatology dataset is 6 class data. No of Attributes = 33
Class 0: Psoriasis- A condition in which skin cells build up and form scales and itchy patches.
Class 1: seboreic dermatitis- A skin condition that causes scaly patches and red skin.
Class 2: lichen planus- An inflammatory condition of the skin and mucous membranes.
Class 3: pityriasis rosea- A skin rash that sometimes begins as a large spot on the chest, belly, or back followed by a pattern of smaller lesions.
Class 4: cronic dermatitis- a rapidly evolving red rash which may be blistered and swollen.
Class 5: pityriasis rubra pilaris- a group of rare skin disorders that present with reddish-orange coloured scaling patches with well-defined borders

#No of ways we can Include this misclassification cost into account while training network:#
1.  Adapting the output of the network: Outputs of the network are changed and appropriately scaled
Minimization of the misclassification costs: The misclassification costs can also be taken in account by changing the error function that is being minimized. Instead of minimizing the squared error, the backpropagation learning procedure should minimize the misclassification costs.
2.  Adapting the learning rate: The idea of this approach is that the high cost examples (that is, examples that belong to classes with high expected misclassification costs) can be compensated for by increasing their prevalence in the learning set. In neural network this can be implemented by increasing learning rate for high cost examples, thus giving them greater impact on the weight changes. To ensure the convergence of the modified backpropagation procedure, the corrected learning rate should also be accordingly


In this implementation, I have used Adapting the learning rate method.
According to classes a Symmetric and Asymmetric Cost matrix is created as mentioned below:
![Symmetric Cost matrix](images%20/symmetric.jpg)
![Asymmetric Cost matrix](images%20/asymmetric.jpg)

Here in symmetric cost matrix cost for misclassification is reduces for the classes they are close to each other and cost increased for classes which has big difference. Asymmetric cost matrix has random cost assigned.
Normal cost matrix has one cost for each pair of misclassification
Misclassification cost is applied in the form of learning rate increase (by constant value c).
Ex: for Symmetric cost metrix
Expected class 1 predict 0, new learning rate will be l_rate + 0.3*C
Expected class 1 predict 4, new learning rate will be l_rate + 0.9*C
Expected class 1 predict 5, new learning rate will be l_rate + 1.2*C
In execution l_rate is 0.5 and C is 0.2
Same rule applied for Asymmetric cost matrix

#Implementation: (Program Flow-You can find relative comments in code in each segment)
1. Read Both training and test datasets
2. Convert data from both dataset to proper format (Attributes to Float, Class value column to Int)
3. Normalize datasets: In this dataset Attribute value vary at large scale so to reduce efforts required to train network I normalized dataset.
4. Create initial network for both the cases.
5. Train network with consideration of Symmetric cost during error propagation, Test trained network (Modified algorithm run)
6. Train network with consideration of Asymmetric cost during error propagation, Test trained network (Modified algorithm run)
7. Train network without consideration of cost during error propagation, Test trained network



##Execution Report##

One Run:
![Run Screeshot](images%20/run.jpg)


Result:
![Result Table](images%20/result.jpg)

(Here in symmetric and asymmetric cost matrix some misclassification class pair has less value and some has more value. The class which are related (close class 0 -1) has less misclassification cost compare to value in Normal matrix. And further classes (big difference in attributes class 0 – 5) have higher misclassification cost)

As you can see in above when we apply cost matrix in training and then test the network we get slightly more accuracy on all the cases. Apart from accuracy total misclassification cost is also decreased in most of the runs. Also, whether its symmetric cost matrix or asymmetric cost matrix, we get improvement in accuracy and total misclassification cost. In some runs we get significant improvement in total misclassification cost as highlighted in above table.
Conclusion: Algorithm is modified to minimize the costs of the errors made. Experiment shows that including misclassification cost in the form of learning rate while training backpropagation algorithm will slightly improve accuracy and improvement in total misclassification cost. There are multiple ways to include misclassification cost and this result may vary depending how you apply that.

#Note#
Modification is done in such a way that the behavior of the modified algorithm remains same to that of the original backpropagation algorithm.
This result will depend on problem dataset you are using and also how you initialize the network in first step.

#References#
**[Research paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.8285&rep=rep1&type=pdf)
**[Machinelearningmastery](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)
**[UCI_Dataset](https://archive.ics.uci.edu/ml/datasets/Dermatology)
