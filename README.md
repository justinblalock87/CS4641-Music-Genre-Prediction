# Music Genre Prediction 

Project for CS4641 Group 40 with Evan Montoya, Ganesh Murugappan, Jason Lee, Charles Luo, and Justin Blalock.

[Video: https://youtu.be/oLWvsiqNnOE](https://youtu.be/oLWvsiqNnOE)

## Introduction
The music streaming industry has exhibited explosive growth with the increasing digitization of entertainment. The leading streaming company, Spotify, has over 360 million users with over 70 million songs in its catalog, and generated over 2.68 billion USD in the 2nd quarter of 2021. Their success is in large part due to both the size of their music catalog and their ability to successfully recommend music; thus, predictive models and relevant data are integral to differentiate themselves from their competition.  

![alt text](https://community.spotify.com/t5/image/serverpage/image-id/76675iC6C745482E249B17/image-size/medium?v=v2&px=400)

## Problem Definition
Using the attributes of songs on Spotify, we would like to predict a song’s genre based on its features. There exists a plethora of available data describing specific songs that Spotify provides and uses for its categorization and recommendation systems, and we plan to use that for our classification task. We would also like to perform evaluation of similarity between songs, in order to categorize them for a potential recommendation system.

## Potential Results and Discussion
We will train our models on the data mentioned and analyze the relationship between the relevant variables, eventually resulting in a probability that a song is in a particular genre. Through this, we hope to find the best model for this song classification task, and ideally achieve a model that reaches the human benchmark for this classification task.

## Data Collection
The data that we will use is sourced from Kaggle under “Dataset of songs in Spotify”. The dataset contains over 10,000 songs per genre. Each data entry contains 22 relevant variables, with 14 describing information (energy, danceability, temp, etc), and the other variables containing other information such as its ID or name. If we need more data, we can also get these same attributes for songs from the Spotify API.

## Methods

### Data cleaning
To clean the data, we began by ensuring that all features had no NaN or Null values in order to ensure consistency across entries. Following this, we eliminated either esoteric genres such as "Ska" or quasi-genres such as "Movie" in order to focus on more relevant genres. Additional cleaning occured in the form of encoding categorical features to numerical ones and eliminating data that contained nulls or nans so that each element in our dataset can be used with confidence. 

### Preprocessing/Feature Engineering
Following this, we began to focus on feature engineering. Firstly, categorical features such as Key or Mode were changed to numbers. Secondly, certain features were just generally irrelevant to our intent of classifying songs into genres based on innate characteristics of the music itself, so categories such as "artist_name", "track_id", and "song_name" simply provided no utility to us. Afterwards, genre labels were changed to a one hot encoding as text labels for most models generally don't work; thus, they must be converted into numerical models. However, an integer representation for each unique genre may result in the model associating an ordinal relationship between genres even though there is no positional significance. Thus, the features were one hot encoded to create n features.

Following these fixes, we then normalized and balanced the dataset. To normalize our data, we used MinMaxScaler to scale and translate every feature between 0 and 1, and they previously had inconsisten scale and distribution. Given that our dataset already consists of over 10,000 songs per genre, the data is already balanced.

#### Data visualization
The cross correlation matrix for the data was constructed using the seaborn package and may be seen below. 

![cs4641 Heat Map](https://user-images.githubusercontent.com/52206987/142122779-c2072db3-94dd-48bf-968d-6102e207908d.png)

We discovered correlations in our data to get some intuition on the data before trying to build models.

We will primarily be using supervised learning, although depending on the success we find with the implementation, we may later expand to unsupervised methods. We will use a variety of supervised methods to allow us to find the algorithms with the highest efficacies, such as, but not limited to, neural networks, decision trees, and logistic regression.

### Dimensionality Reduction
After our data was fully cleaned, we found through a PCA analysis that we only needed 12 out of our 14 features to retain 99% of the variance in our data. Thus, we were able to further eliminate two unnecessary features.

![scree (1)](https://user-images.githubusercontent.com/52206987/142089842-aba671d9-b6b4-4809-9268-a90d47384b2f.jpeg)

## Supervised Methods

### Random Forest
Following this, we were ready to begin applying the ML algorithms themselves. For the midpoint, we focused on implementing one Supervised Learning algorithm: RandomForestClassifier. To do this, we capitalized on the sklearn library amd applied a RandomForestClassifier to our dataset, which yielded an accuracy of 45.9%. In addition, we had an f1 score of .44. While not ideal, it demonstrates potential which we will expand upon by the final report. 

![cs4641randomforest](https://user-images.githubusercontent.com/52206987/142121286-487eee83-c7db-4e5c-b133-b78d1800a92e.png)

At the moment, our current accuracy has only been capable of reaching 45%. While an improvement over random selection, we are currently limited by the features that were extracted from the spotify API. To make our model more accureate, we intend to do a spectrogram analysis to ideally extract more information to classify songs. This would consist of normalizing the spectrogram against volume and combining it with the existing spotify data.

## Timeline and Responsibilities
We will have our data cleaned and processed by October 25th. We intend to have the implementation of our first approach by November 8, 2021. We will aim to be able to compare the effectiveness of several approaches by November 28, 2021. We will then have sufficient time to consolidate and write up our results before the deadline. These deadlines will allow us the necessary time to elaborate and improve upon the project before the respective deadlines. With respect to the distribution of our responsibilities, we intend to have each individual focus on a particular approach. Justin Blalock will primarily be responsible for data processing and managing the GitHub. Charlie and Jason will be responsible for working on analyzing actual audio samples. Ganesh will be in charge of the neural networks. Evan will work on the Decision Tree and dimensionality reduction. These responsibilities are subject to change.

## References
- G. Tzanetakis and P. Cook. Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5):293–302, July 2002.
- Hareesh Bahuleyan. Music genre classification using machine learning techniques. CoRR, abs/1804.01149, 2018. 
- Mingwen Dong. Convolutional neural network achieves human-level accuracy in music genre classification. CoRR, abs/1802.09697, 2018. 
- Pelly, L. (2017). The Problem with Muzak: Spotify’s bid to remodel an industry. The Baffler, 37, 86–95. http://www.jstor.org/stable/26358588
