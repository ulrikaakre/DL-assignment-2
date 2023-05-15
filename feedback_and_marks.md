# ST456 Assignment 2 - Provisional feedback and marks

## Questions

### P1-1
The padding should have been on the convolutions. The specification of the max pooling on the second branch is wrong as you forgot to specify the keywords (look at the output size, is not what you expect!). Therefore you get way more parameters than expected.

### P1-2
You were required to use squeezing to remove the channel dimension. Using as input shape (80, 80, 1) defeated the point and forced you to use Conv2D and then the reshape layer (which was not requested). Conv1D would have solved it.

Wrong activation function on convolutions.

### P2

The proposed architecture is a combination of CNN layers into a single branch, with varying number of units and kernel size, followed by RNN layers. Excellent rationale behind the proposed model and the choice of parameters (model and training), and on the observed behaviour of your model. Accuracy is higher than 85% from the half of the training time and stable in the remaining time. There is still room for improvement, especially for reducing overfitting over the validation data.

## Marking scheme

| Problem breakdown | Max points | Your points |
|:--------|-----------:|-----------:|
| Total | 100 | 70 |
