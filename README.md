# Music Genre Prediction 

Project for CS4641 Group 40 with Evan Montoya, Ganesh Murugappan, Jason Lee, Charles Luo, and Justin Blalock.

[Video: https://youtu.be/oLWvsiqNnOE](https://youtu.be/oLWvsiqNnOE)

## Introduction
The music streaming industry has exhibited explosive growth with the increasing digitization of entertainment. The leading streaming company, Spotify, has over 360 million users with over 70 million songs in its catalog, and generated over 2.68 billion USD in the 2nd quarter of 2021. Their success is in large part due to both the size of their music catalog and their ability to successfully recommend music; thus, predictive models and relevant data are integral to differentiate themselves from their competition.  

![alt text](https://community.spotify.com/t5/image/serverpage/image-id/76675iC6C745482E249B17/image-size/medium?v=v2&px=400)

## Problem Definition
Using the attributes of songs on Spotify, we would like to predict a song’s genre based on its features. There exists a plethora of available data describing specific songs that Spotify provides and uses for its categorization and recommendation systems, and we plan to use that for our classification task. We would also like to perform evaluation of similarity between songs, in order to categorize them for a potential recommendation system.

## Data Collection
The data that we will use is sourced from Kaggle under “Dataset of songs in Spotify”. The dataset contains over 42,000 songs. Each data entry contains 22 relevant variables, with 11 describing information (energy, danceability, temp, etc), and the other variables containing other information such as its ID or name. If we need more data, we can also get these same attributes for songs from the Spotify API.

## Methodology
We will primarily be using supervised learning, although depending on the success we find with the implementation, we may later expand to unsupervised methods. We will use a variety of supervised methods to allow us to find the algorithms with the highest efficacies, such as, but not limited to, neural networks, decision trees, and logistic regression.

## Potential Results and Discussion
We will train our models on the data mentioned and analyze the relationship between the relevant variables, eventually resulting in a probability that a song is in a particular genre. Through this, we hope to find the best model for this song classification task, and ideally achieve a model that reaches the human benchmark for this classification task.

## Timeline and Responsibilities
We will have our data cleaned and processed by October 25th. We intend to have the implementation of our first approach by November 8, 2021. We will aim to be able to compare the effectiveness of several approaches by November 28, 2021. We will then have sufficient time to consolidate and write up our results before the deadline. These deadlines will allow us the necessary time to elaborate and improve upon the project before the respective deadlines. With respect to the distribution of our responsibilities, we intend to have each individual focus on a particular approach. Justin Blalock will primarily be responsible for data processing and managing the GitHub. Charlie and Jason will be responsible for working on analyzing actual audio samples. Ganesh will be in charge of the neural networks. Evan will work on the Decision Tree and dimensionality reduction. These responsibilities are subject to change.

## References
- G. Tzanetakis and P. Cook. Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5):293–302, July 2002.
- Hareesh Bahuleyan. Music genre classification using machine learning techniques. CoRR, abs/1804.01149, 2018. 
- Mingwen Dong. Convolutional neural network achieves human-level accuracy in music genre classification. CoRR, abs/1802.09697, 2018. 
- Pelly, L. (2017). The Problem with Muzak: Spotify’s bid to remodel an industry. The Baffler, 37, 86–95. http://www.jstor.org/stable/26358588
