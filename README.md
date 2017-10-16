# Backpropagation-in-python
Backpropagation implementation in python. Misclassification cost is referred while training network.

Dermatology dataset is used to train a backprop network here. Dermatology dataset is 6 class data.

No of Attributes = 33

Class 0: Psoriasis- A condition in which skin cells build up and form scales and itchy, dry patches. Mainly on Scalp
Class 1: seboreic dermatitis- A skin condition that causes scaly patches and red skin, mainly on the scalp.
Class 2: lichen planus- An inflammatory condition of the skin and mucous membranes.
Class 3: pityriasis rosea- A skin rash that sometimes begins as a large spot on the chest, belly, or back followed by a pattern of smaller lesions.
Class 4: cronic dermatitis- a rapidly evolving red rash which may be blistered and swollen.
Class 5: pityriasis rubra pilaris- a group of rare skin disorders that present with reddish-orange coloured scaling patches with well-defined borders

No of ways we can Include this misclassification cost into account while training network:

Adapting the output of the network: Outputs of the network are changed and appropriately scaled

Minimization of the misclassification costs: The misclassification costs can also be taken in account by changing the error function that is being minimized. Instead of minimizing the squared error, the backpropagation learning procedure should minimize the misclassification costs.

Adapting the learning rate: The idea of this approach is that the high cost examples (that is, examples that belong to classes with high expected misclassification costs) can be compensated for by increasing their prevalence in the learning set. In neural network this can be implemented by increasing learning rate for high cost examples, thus giving them greater impact on the weight changes. To ensure the convergence of the modified backpropagation procedure, the corrected learning rate should also be accordingl

In this implementation, I have used Adapting the learning rate method.

According to classes a sample Cost matrix is created as mentioned below:

        Class 0  Class 1 Class 2 Class 3 Class 4 Class 5
        
Class 0    0       0        1       1     1       1.5

Class 1    0       0        1       1     1       1.5

Class 2    1       1        0       0     0        1

Class 3    1       1        0       0     0        1

Class 4    1       1        0       0     0        1

Class 5   1.5     1.5       1       1     1        0

Misclassification cost is applied in the form of learning rate increase (by constant value c).

Ex:
Expected class 1 predict 0, new learning rate will be l_rate + 0*C
Expected class 1 predict 4, new learning rate will be l_rate + 1*Cs
Expected class 1 predict 5, new learning rate will be l_rate + 1.5*C

In execution l_rate is 0.5 and C is 0.2

Implementation: (Program Flow-You can find relative comments in code in each segment)

1. Read Both training and test datasets
2. Convert data from both dataset to proper format (Attributes to Float, Class value column to Int)
3. Normalize datasets: In this dataset Attribute value vary at large scale so to reduce efforts required to train network I normalized dataset.
4. Create initial network for both the cases.
5. Train network with consideration of cost during error propagation 6. Test trained network
7. Train network without consideration of cost during error propagation

Conclusion:

With Adaptive learning rate, we are trying to minimize misclassification cost by giving the higher weight to high-cost examples during training network. Once network is trained we get more accuracy on test dataset when we considered misclassification cost into account then we train without considering cost.

In all the cases, no matter how we try to train network (with different iteration count, or different no of nodes in hidden layer) we get slightly better result in accuracy on test data with cost consideration.

This accuracy improvement will vary depending on dataset you are using for backpropagation along with how you initialize network in the first step.
