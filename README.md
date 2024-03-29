# Music Genre Prediction 

Project for CS4641 Group 40 with Evan Montoya, Ganesh Murugappan, Jason Lee, Charles Luo, and Justin Blalock.

[Final Video: https://youtu.be/yaMitUbwBP4](https://youtu.be/yaMitUbwBP4)

## Introduction
The music streaming industry has exhibited explosive growth with the increasing digitization of entertainment. The leading streaming company, Spotify, has over 360 million users with over 70 million songs in its catalog, and generated over 2.68 billion USD in the 2nd quarter of 2021. Their success is in large part due to both the size of their music catalog and their ability to successfully recommend music; thus, predictive models and relevant data are integral to differentiate themselves from their competition.  

![alt text](https://community.spotify.com/t5/image/serverpage/image-id/76675iC6C745482E249B17/image-size/medium?v=v2&px=400)

## Problem Definition
Using the attributes of songs on Spotify, we would like to predict a song’s genre based on its features. There exists a plethora of available data describing specific songs that Spotify provides and uses for its categorization and recommendation systems, and we plan to use that for our classification task. We would also like to perform evaluation of similarity between songs, in order to categorize them for a potential recommendation system.

## Data Collection
The data that we will use is sourced from Kaggle under “Spotify Tracks DB”. The dataset contains about 10,000 songs per genre with 26 genres (232,725 total songs) and was obtained using the Spotify API. Each data entry contains 18 relevant variables, with 14 core features including energy, danceability, duration, etc. If we need more data, we can also get these same attributes for songs from the Spotify API.

## Methods Part 1

### Data cleaning
To clean the data, we began by ensuring that all features had no NaN or Null values in order to ensure consistency across entries. Following this, we eliminated either esoteric genres such as "Ska" or quasi-genres such as "Movie" in order to focus on more relevant genres. Additional cleaning occured in the form of encoding categorical features (key and time signature) to numerical ones and eliminating data that contained nulls or nans so that each element in our dataset can be used with confidence. 

### Preprocessing/Feature Engineering
Following this, we began to focus on feature engineering. Firstly, categorical features such as Key or Mode were changed to numbers. Secondly, certain features were just generally irrelevant to our intent of classifying songs into genres based on innate characteristics of the music itself, so categories such as "artist_name", "track_id", and "song_name" simply provided no utility to us. Afterwards, genre labels were changed to a one hot encoding as ordinal encoding would not be a good representation of the genres. An integer representation for each unique genre may result in the model associating an ordinal relationship between genres even though there is no positional significance. Thus, the features were one hot encoded to create n features.

Following these fixes, we then normalized and balanced the dataset. To normalize our data, we used MinMaxScaler to scale and translate every feature between 0 and 1, as they previously had inconsistent scale and distribution. Given that our dataset already consists of over 10,000 songs per genre, the data is already balanced.

### Data visualization
The cross correlation matrix for the data was constructed using the seaborn package and may be seen below. 

![cs4641 Heat Map](https://user-images.githubusercontent.com/52206987/142122779-c2072db3-94dd-48bf-968d-6102e207908d.png)

We discovered correlations in our data to get some intuition on the data before trying to build models.

### Dimensionality Reduction
After our data was fully cleaned, we found through a PCA analysis that we only needed 12 out of our 14 features to retain 99% of the variance in our data. Thus, we were able to further eliminate two unnecessary features.

![scree (1)](https://user-images.githubusercontent.com/52206987/142089842-aba671d9-b6b4-4809-9268-a90d47384b2f.jpeg)

## Preliminary Results

### Random Forest
Following this, we were ready to begin applying the ML algorithms themselves. For the midpoint, we focused on two supervised learning algorithms, one of which is RandomForestClassifier. To do this, we capitalized on the sklearn library amd applied a RandomForestClassifier to our dataset, which yielded an accuracy of 45.9%. In addition, we had an f1 score of .44. While not ideal, we will expand on this and make random forest better in part 2 (the next major section).

![cs4641randomforest](https://user-images.githubusercontent.com/52206987/142121286-487eee83-c7db-4e5c-b133-b78d1800a92e.png)

We can visualize the mistakes our model is making with a confusion matrix.

![Confusion Matrix](confusionmatrix_image.png)

As we can see, the model has difficulty telling the difference between similar genres like Hip-Hop and Rap or Dance and Pop. This is expected and this combined with the f1 score shows that perhaps our model could use more complexity because its precision and recall is struggling.

### Neural Network
We created a neural network with 4 hidden layers all using ReLU as an activation function except the output layer which uses softmax.

![Neural Network](Unknown-3.png)

Our highest accuracy using this method was 45.2%. For both supervised methods, the models aren't doing terribly. There are 14 genres which means random guesses would yield an accuracy of ~7%. We suspect the main issue to be the lack of information provided by our features. Many of the features like 'dancability' were abitrary and uninformative. Because of this, we decided to begin analyzing raw audio files instead. 

## A New Dataset
We switched to the GTZAN dataset (http://marsyas.info/downloads/datasets.html). This dataset consists of 10 genres, 100 audio files per genre, and 30 seconds for each audio file. From here, we obtained a dataset with low-level features such as chroma features and spectral centroid information extracted from the audio files (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification). Additionally, our dataset was expanded such that each 30 second audio clip was split into ten 3 second subsections. This ultimately increases the amount of data by a factor of ten. More data will allow our models to be trained with greater accuracy. In sum, the difference between this dataset and our previous dataset is that this one has features that literally extract audio signals whilst the other dataset had quasi-features that were moreso engineered into being.

General Data Visualization             |  Data Features
:-------------------------:|:-------------------------:
![caplot_of_data](catplot.png)  |  <img src="box_features.png" height="300" width="600" align="center">

## Methods Part 2

### Data cleaning, Preprocessing/Feature Engineering, and Dimensionality Reduction (Again)

In a similar manner to our previous dataset, we first cleaned the data and then performed some feature engineering. We got rid of the filename and length features, as they were all the same length and filename is a string value. Furthermore, we ensure that there were no null or NA values. Fortunately, the data was already balanced, so from here we simply had to normalize the data. Once again, we used MinMaxScaler to scale and translate the features between 0 and 1, as this controls for inconsistent scale and distrubtion. Finally, we encoded the genres since they were strings.

Once everything was normalized and controlled, we applied PCA on the data as there were 57 features. Setting PCA to retain 99% variance reduced the total number of features to 47 (43 in the video but the final model uses 47 because of slight changes in the feature engineering explained above).

### New Neural Network

We began by creating a new neural network.

![cs4641cnnTraining](https://user-images.githubusercontent.com/52206987/144925992-783df712-91e6-41b1-a15a-1bb4b8e70a36.png)

To tune the hyperparameters of our model, we used keras-tuner and we tuned the number of nodes in the first and second layers of the network as well as the learning rate.

We then trained the model using 70% of the data, used 10% as a means of validation, and tested it on 20% of the data. This yielded a final accuracy of 91%.

![model_history_plot](model_history.png)

Additionally, we applied random forest to our model to achieve 82% accuracy and an f1 score of .82. We tuned the hyperparameter for the depth of the tree and saw diminishing returns after a depth of 30.

Here is a view of the forest. While the data is illegible, this image shows the complexity of the classification task.

![new_data_forest](new_data_forest.png)

Additionaly, we applied logistic regression to our new dataset and obtained 67% accuracy.

## Discussion
Changing our dataset drastically improved the accuracy of our model. When working with GTZAN, our neural network was by far our most accurate model, yielding an accuracy of 91%. Our random forest model produced results of 82% accuracy, with logistic regression lagging behind slightly at 67%. On one hand, the random forest had great precision and recall. But ultimately, we would consider our neural network to be successful at its target task of accurately classifying the genre of a song. Additionally, we tuned the hyperparameters as mentioned in the last section to obtain optimal models. Tuning the hyperparameters allows us to, of course, select the parameters that perform best for the model and data. It also helps to prevent overfitting, which was an issue for us at first. At an accuracy of 91%, it is possible that our model would see some success for assisting in music recommendation algorithms. 

## Conclusion
In future studies, it could be useful to work on classifying subgenres or increasing the size of the dataset. Furthermore, there is great promise in converting audio files to Mel spectrograms and feeding those images to convolutional neural networks. Spectrogram analysis yields a lot of information. Finally, compared to other methods of genre classification, we believe our method is among the most accurate and reliable since we used features that closely reflected raw audio and used a network that mirrored this complexity.

## Timeline and Responsibilities
We will have our data cleaned and processed by October 25th. We intend to have the implementation of our first approach by November 8, 2021. We will aim to be able to compare the effectiveness of several approaches by November 28, 2021. We will then have sufficient time to consolidate and write up our results before the deadline. These deadlines will allow us the necessary time to elaborate and improve upon the project before the respective deadlines. With respect to the distribution of our responsibilities, we intend to have each individual focus on a particular approach. Justin Blalock will primarily be responsible for data processing and managing the GitHub. Charlie and Jason will be responsible for working on analyzing actual audio samples. Ganesh will be in charge of the neural networks. Evan will work on the Decision Tree and dimensionality reduction. These responsibilities are subject to change.

## References
- G. Tzanetakis and P. Cook. Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5):293–302, July 2002.
- Hareesh Bahuleyan. Music genre classification using machine learning techniques. CoRR, abs/1804.01149, 2018. 
- Mingwen Dong. Convolutional neural network achieves human-level accuracy in music genre classification. CoRR, abs/1802.09697, 2018. 
- Pelly, L. (2017). The Problem with Muzak: Spotify’s bid to remodel an industry. The Baffler, 37, 86–95. http://www.jstor.org/stable/26358588
