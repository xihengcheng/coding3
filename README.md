Input words to produce the corresponding verse system


In this project, I hope to build a machine learning model, and the final effect is that the user can enter a random keyword and generate a poem with this keyword. But I know that doing so requires a relatively large database model and a long training time to achieve the desired results.


At the beginning of my project, I referenced a machine learning model for randomly generating fake news on github, whose main function was to implement a text generation system. It utilizes a recurrent neural network (RNN), specifically a short term memory network (LSTM), to predict and generate new text. (https://github.com/NeuralNine/youtube-tutorials/blob/main/Text%20Generation%20AI%20-%20Next%20Word%20Prediction/Text%20 Generation%20AI.ipynb)


On the basis of this model, I want to change its theme and become a poetry generation system.
I took advantage of the common Python libraries for this machine learning model, such as random, numpy, pandas, tensorflow, nltk, etc., and retained the LSTM model for processing sequence data.
At the same time, the RegexpTokenizer in the model is used for word segmentation and word extraction, the LSTM layer is used to capture the time dependence of sequence data, and the Dense layer and softmax activation function are used to output the probability distribution of the next word.


I encountered a series of challenges during the debugging process. First, the most important one was that I changed the database. When I loaded the poetry data set, the system memory was often insufficient during the training process. Due to the limitation of computer performance, I had to intercept part of the poetry content of the database for training. Although it could achieve the general effect, the fluency of the poetry 
would be lost to some extent.


Reduce memory pressure by adding.head() to capture part of the database sample



At the same time, in the process of testing, I found that the produced poems would produce more repeated words, which I wanted to avoid. Therefore, I hope to improve the generation strategy, check the candidate words during generation, and select other candidate words if they are newly generated words.



I constantly adjust the size of the imported database to make the learning results as accurate as possible.
In the course of the test, I found that the resulting lines had numbers in them. To avoid this, I tried to add a filter step, filtered_candidates, to the predict_next_word function to exclude numbers from the candidates. To avoid producing duplicate words, I added the if function to avoid selecting the most recently added words


I modified the LSTM layer with 256 units per layer, trained 5 epochs with a batch size of 32, and changed the long short-term memory network in an attempt to increase the depth and complexity of the model to better capture complex patterns and dependencies in sequence data.

Due to the limitation of computer performance, the poems I produced were still not smooth enough. When I had more memory space, I increased the training database and expanded the number of training times, which should greatly increase the accuracy of training




