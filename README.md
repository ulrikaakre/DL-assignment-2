# Assessment 2: Music genre classification

## Data

Download the [training and validation data](https://drive.google.com/drive/folders/154dxA9DPaEUW_QbCuW8e-eA5xYd73CQL?usp=sharing). It is recommended you download the data folders and then place them in the Google Drive associated to your Colab account. Then, once you mount your Drive on Colab, you can easily access them with 

`train_dataset = tf.data.Dataset.load('<path>')`

Before running the model, you need to pre-batch the data using the following commands:

`batch_size = 128`

`train_dataset_batch = train_dataset.batch(batch_size)`

and similarly for the valiation set.

The data represent the log-transformed Mel spectrograms derived from the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The original GTZAN dataset contains 30-seconds audio files of 1,000 songs associated with 10 different genres (100 per genre). We have reduced the original data to 8 genres (800 songs) and transformed it to obtain, for each song, 15 log-transformed Mel spectrograms. Each Mel spectrogram is an image file represented by a tensor of shape (80, 80, 1) which describes time, frequency and intensity of a song segment. The training data represent 80% of the total number of data points.

# P1 - Parallel CNNs and RNNs

## P1.1

The goal is to train a CNN based classifier on the Mel spectrograms to predict the corresponding music genres. Implement the shallow parallel CNN architecture according to the following specification:

- First parallel branch:
  1. one convolutional layer processing the input data with 3 square filters of size 8, padding and leaky ReLU activation function with slope 0.3.
  2. one pooling layer which implements Max Pooling over the output of the convolutional layer, with pooling size 4.
  3. a layer flattening the output of the pooling.

- Second parallel branch:
  1. one convolutional layer processing the input data with 4 square filters of size 4, padding and leaky ReLU activation function with slope 0.3.
  2. one pooling layer which implements Max Pooling over the output of the convolutional layer, with size 2.
  3. a layer flattening the output of the pooling..

- Merging branch:
  1. a layer concatenating the outputs of the two parallel branches.
  2. a dense layer which performs the classification of the music genres using the approppriate activation function.

Use an appropriate evaluation metric and loss function. Optimise using a mini-batch stochastic gradient descent algorithm. Train for 50 epochs.
Plot the loss function and the accuracy versus the number of epochs for the train and validation sets. Your model should achieve at least 70% accuracy on the validation set.

## P1.2

The goal is to train a CNN-RNN based classifier on the Mel spectrograms to predict the corresponding music genres. First, you need to reduce the dimensionality of your dataset by running the following code:

`def reduce_dimension(x, y):`

`  return tf.squeeze(x, axis=-1), y`

`train_dataset_squeeze = train_dataset.map(reduce_dimension)`

and similarly on the validation set. Then, batch the resulting train and validation sets with batch size of 128.

Second, implement the following CNN-RNN architeture:
1. a convolutional layer with 8 square filters of size 4.
2. a max pooling layer that halves the dimensionality of the output.
3. a convolutional layer with 6 square filters of size 3.
4. a max pooling layer that halves the dimensionality of the output.
5. an LSTM layer with 128 units, returning the full sequence of hidden states as output.
6. an LSTM layer with 32 units, returning only the last hidden state as output.
7. a dense layer with 200 neurons and ReLU activation function.
8. a layer dropping out 20% of the neurons during training.
9. a dense layer which outputs the probabilities associated to each genre.

Use an appropriate evaluation metric and loss function. Optimise using a mini-batch stochastic gradient descent. Train for 50 epochs.
Plot the loss function and the accuracy on the train and validation sets over the epochs. Your model should achieve at least 50% accuracy on the validation set.

# P2 - Achieving higher accuracy

Considering the results you obtained in P1, implement a neural network architecture that achieves at least 85% accuracy on the validation set, over 50 epochs. To do so, you are restricted to some of the following modifications:

- data augmentation techniques.
- different CNN architectures (with or without parallel branches).
- different RNN architectures.
- different mix of CNNs and RNNs.
- different optimisers or learning rate choices.

Present and describe the architecture you have chosen and justify the rationale behind it. Plot training and validation loss and accuracy over 50 epochs.

## Marking scheme

| Problem | Full points | Max points |
|:--------|-----------:|-----------:|
| P1-1  | 25 | 25  |
| P1-2  | 25 | 25  |
| P2  | 50 | 50  |
| Total | 100 | 100 |

## Submission date and guidelines

1. Solution deadline: **Thursday 30 March, 5 pm**.
2. Document your Python code with proper sections for each questions and provide all the comments necessary to understand your code.
3. Upload your Python code (including discussion/explanation for each question) and additional files (if any) into the GitHub repository automatically created with the link sent for the assignment. 
4. There is no need for uploading training and testing data, **as these datasets cannot be changed**.
5. **IMPORTANT**: for completing your submission, go to [Moodle (Assessment 2 section)](https://moodle.lse.ac.uk/mod/assign/view.php?id=1169778) and **provide a file (either pdf/docx/txt) with a link to your GitHub repository** (this must be done by the deadline). Your submission should be titled clearly with your LSE Candidate Number (your 5-digit candidate number available on LfY). Please make sure that your submission is **not in draft mode**.

