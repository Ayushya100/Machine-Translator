# Machine Translator using Model Subclassing
## Overview about the repository
The CSV file named english_hindi.csv is a file that contains the english sentences and their corresponding hindi sentences. This csv file contains around 40k instances of English-Hindi
sentences.

The file named as Language Translator.ipynb is a python script that contains the code for building the model i.e. to create the model and train the model. and the file named as
Translator Implmentor.ipnb is a python script that contains the code to load the model i.e. in this script we are loading our save model and weights to repeatdly use it without
requiring us to first train the model each time when we want to use it.

## Introduction
The basic idea behind the development of MT system is to bridge the gap between people who speak two different languages and make it easier for them to communicate with each other, by
converting a message of one person into a language that can be understood by another and vice-versa.

A Machine Translation model takes a string written in a particular language as input and generates a corresponding string in another langauge.

In Machine Translation, the English language, i.e. the language of the sentence to be translated is called source language. The Hindi language, i.e. the language of the translated
sentence, is called the target language.

The overall approach of using the Encoder-Decoder model in this project is to train the Encoder-Decoder model by first converting the input to the corresponding context vector and then
getting the output text from the generated context vector.

There are a number of steps that needs to be completed for developing a machine translation model. These steps are describes as follows:

### A. The Dataset
The dataset that is used for training the model is a CSV file containing English sentences and their corresponding Hindi sentences. This dataset named english_hindi.csv is attached in
this repository. This dataset contains 40k instances of English-Hindi sentences, once the dataset is loaded then further data preprocessing is to be performed.

### B. Data Preprocessing
Data Preprocessing is a very vital step in any machine learning activity. Preprocessing is very important as a result it completely directs ML activities and makes the additional
processes less complicated.

The preprocess task starts initially by changing the given input sentence into lowercase, then further areas, single quotes, punctuations, and different stop words are removed from the
sentence. Once this is done English and Hindi numbers are removed from the sentence. In the end, 'start_' and '_end' tokens are added at the start and end of the sentences.

Once all the sentences come to a simpler form, tokenization is the next step that needs to be performed. Tokenization essentially splits sentences, paragraphs, or an entire text
document into smaller unit’s i.e. individual words. Each of these individual words is called a token. Tensorflow  offers a simple function named ‘Tokenizer’ to get the individual
tokens. All these tokens are stored in a list of texts

When feeding words to a machine translation model, words need to be converted to a numerical representation. ‘texts_to_sequences’ is a function that takes words as a input and returns
their corresponding numerical representations. In the end before passing the list to the model every sequence should be of the same length. ‘pad_sequences’ is a function, which ensures
that every sequence in a list has the same length.

Tokens is then splited in the relation of 80:20 in the list of source and target tokens. Where 80 percent of the data is used for training purpose and the rest 20 percent is used for
testing purpose.

### C. Encoder-Decoder Architecture
THe Encoder-Decoder architecture is a recurrent neural network that is used for addressing sequence-to-sequence problems, therefore it is also known as seq2seq architecture. This
architecture is widely used for building machine translation applications.

A machine translation model works by, first, consuming the words of the source language sequentially, and then, sequentially predicting the corresponding words in the target language.
However, under the hood, it is actually two different models, an encoder and a decoder.

The Encoder & Decoder models that we have created are sub-classed models. Model Sub-classing is used for custome-designing of feed-forward mechanism of deep neural network. It contains
two important functions, they are '__init__' and 'call'. In the '__init__' method, we define all our tf.keras.layers as well as any custom implemented layers. In the 'call' method we
call all these layers that are defined in the '__init__' method to accomplish feed-forward propagation.

In Natural Lnaguage Processing, the attention mechanism outperforms the encoder-decoder machine translation system.

#### 1. The Encoder
The Encoder takes the numerical representation of English words a input and produces a corresponding representation known as context vector.

Some important attributes of the dataset are the average number of words, the average length of the sentence and the size of the vocabulary. These parameters are required to define the
input layer of the Encoder.

The encoder is a sequential model that goes from one input to the next while creating an output at each time step. At time step 1 the first word is processed, at time step 2 the
second word is processed and so on upto the last word. In our translator, the encoder model is made up of the GRU layers, which stands for Gated Recurrent unit.

The GRU layer absorbs the input, absorbs the starting state and produces the new state at each time step as an output. This GRU layer continuesto work in this manner until it reaches
the end of the sentence. The prior step's hidden state serves as a memory of what the model has previosly observer. These hidden states are calculated using the GRU model's internal
parameters, which are learned during model training. The final state of the  GRU layer is a context vector, which will be sent to the decoder as inputs later.

#### 2. Bahdanau Attention
Bahdanau et al. proposed an interesting mechanism that learns to align and translate jointly. It is likewise referred to as Additive interest because it plays a linear mixture of
encoder states and decoder states.

#### 3. The Decoder
The decoder consumes the context vector as an input and produces probabilistic predictions for each time step. The word for a given time step is selected as the word with the highest
probability. Though the inputs to the encoder are ones and zeros, the decoder produces continuous probabilistic outputs. The Decoder is implemented identically to the Encoder using 
the keras GRU layer.

To get the output sentence, repeat the context vector N times, for example, to get a ten-word Hindi sentence, repeat the context vector 10 times. We use the RepeatVector layer offered
by keras for this. We may use the RepeatVector layer to repeat an input for a set amount of times.

Using the RepeatVector layer, we define the decoder input by repeating the encoder state for a specified number of times. Now we'll put together the Decoder GRU layer. All of the
decoder's GRU layer outputs are required in Decoder because each of those GRU outputs is ultimately used to forecast the proper Hindi word for each decoder location.

### D. The Training Phase
The source and target datasets are input into the encoder layer after the dataset has been preprocessed to prepare the context vectors from the senteces. The decoder is then given the
encoder output, the encoder hidden states, and the decoder input. The predictions and hidden states are returned by the decoder. The hidden state of the decoder is then passed back into
the odel, and the loss is calculated using the predictions.

We use GRU and an attention mechanism in our model design. We also include a dense layer that serves as a link between the encoder and the decoder. We employed Teacher forcing to determine
the decoder's next input. The next input to the decoder is determined by teacher forcing. The target word is sent as the next iput to the decoder using the teacher forcing technique.

We are utilizing a dataset size 40000 for training our model in the ratio of 80:20, which means that we are using 80% of the data for training. We are training our model through google
colab to conduct 20 epochs. The neural network takes about an hour to train for each epoch because we used GPU.

As the dataset and the number of epochs grow larger, the model's prediction accuracy improves. However, there is a limit to how many epochs can be used; once this limit is reached,
increasing the number of epochs will not improve accuracy. The evaluation function is quite identical to the training loop, with the exception that we do not employ teacher forcing.

### E. Model Saving
Saving the sub-classed models is quite difficult comparing to the other two ways of model building i.e. sequential and functional api's, because a sub-classed models cannot be saved
directly to hdf5 or joblib.

To save the sub-classed model, first we need to save the raw models i.e. an untrained model and then we need to save the weights of the model. Later we can use these saved models and
weights where ever we want it to implement our model.

Here in this repository file named as Language Translator.ipynb is a python script that contains the code for building the model i.e. to create the model and train the model. and the
file named as Translator Implmentor.ipnb is a python script that contains the code to load the model i.e. in this script we are loading our save model and weights to repeatdly use it
without requiring us to first train the model each time when we want to use it.

### F. Result
When we give the model some sentences to translate, at least six to seven sentences out of ten mathces the expected result, So we can conclude that our system is 70% accurate.

## Future Work
As a future work we can use this model with computer vision and with speech to text so that users with some diability can speak or can show the image in which somthing prewritten needs
to be translated.


