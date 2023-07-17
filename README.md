Recommendation System <!--S'ha de tornar a posar pq el primer no es guarda al .md-->
===
push test
Final Project for the UPC Artificial Intelligence with Deep Learning Postgraduate Course, Spring 2023.

* Authors: [Núria Gonzalez](mailto:nugobo1981@gmail.com), [Jaume Betriu](betriutortjaume@gmail.com), [Miquel Albó](mailto:miquel.albo@estudiantat.upc.edu)
* Team Advisor: [Paula Gomez Duran](paulagomezduran@gmail.com)
* Date: July 2023

## Table of Contents (Provisional)

* [Introduction](#intro)
* [Overview](#overview)
    * [Setup & Usage](#setup_usage)
    * [Architectures](#architecture)
      * [Random](#architecture)
      * [Popularity](#popularity)
      * [Collaborative Based Filtering](#collab_filter)
      * [Content Based Filtering](#content_filter)
      * [Techniques explored](#tech_exp)
    * [Dataset](#dataset)
      * [Preprocessing](#data_features)
      * [Negative sampling](#data_neg)
* [Metrics for evaluation](#metrics)
    * [Hit rate](#hitrate)
    * [Intralist (NDCG)](#intralist)
    * [Coverage](#coverage)
* [CF Models Exploration:: Understanding and Comparison](#explore)
    * [Random](#random)
    * [Popularity](#popularity)
    * [Matrix Factorizacion and Factorization Machine Model (FM)](#fm_model)
    * [Graphical Convolutional Network Model (GCN)](#gcn_model)
    * [Bias problems](#bias)
        * [Cold start](#cold_start)
        * [Filter bubbles](#filbub)
        * [Scalability](#scale)
        * [Non-interpretable results](#nonint)
        * [Data hungry](#datahung)
    * [Deep Learning Models (DL)](#deep_model)
        * [Our proposal](#dl_model)
* [Results](#results)
* [Conclusions](#conclusion)
* [References](#references)


# Introduction <a name="intro"></a>
Recommender systems filter information to avoid overload because of the amount of data generated related to users' interests. Retrieval algorithms used in systems such as Google, Altavista, Grouplens or Amazon are some examples of mapping the available content to the preferences of the users based on their observed behavior about an item.

Our project explores the most mature techniques for the development of decision strategies for users in complex information environments. The project's focus is centered on understanding how they work, the advantages and disadvantages of each model, and alternative proposals.

You will find the papers consulted to carry out the project in the links at the end of the document.


# Setup & usage (keywords) <a name="setup_usage" ></a>

Firstly, clone this repository to your local machine.

## Setup

To run the project, you have two options: Using `(1) Conda Environment` or perform a `(2) manual installation`:

### 1. Conda Environment

If you have Anaconda or Miniconda installed, you can import the `recommender_system_conda.yml` file to automatically set up the environment with all the required dependencies. This eliminates the need to install the `requirements.txt` file.

#### Importing the Conda environment
To import the Conda environment, execute the following command in your terminal:
```bash
conda env create -f recommender_system_conda.yml
```
Next, activate the environment as you normally would:

```bash
conda activate recommender_system_final
```


### 2. Manual Installation

The manual installation consists on:

1. Install the required packages specified in the `requirements.txt` file using `pip`. 
2. Ensure compatibility with your `CUDA` version by modifying the version specified in the `requirements.txt` file to match that of your local machine.
    
    This is the line you have to modify on `requeriments.txt`:
    ```python=1
    torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    > Example with torch version 2.0.0 and cuda 11.8

## Usage

To enhance the user experience and minimize waiting times, the repository is executed in two stages:
- Data preprocessing
- Model training

The first step will be to create a data folder inside the rpository. The name of the folder **must be 'data'**. Inside the folder the user must place the dataset **movies_samples.csv**.

To run the preprocessing of the data:

```bash
python preprocessing_main.py
```

This will create 5 different csv files in the data folder.

At this point you are now able to train the models. To do so run: 

```bash
python main.py --model <model_name>
```

Replace the variable `<model_name>` with the name of the model you want to train and evaluate. This are the available models:
- `random`
- `popularity`
- `deep`
- `residual`
- `compact`
- `fm`
- `abs_popularity`

A pop up will appear showing the loss function of the train and validation datasets during the training. To get the test metrics close the pop up and the code will continue running.

# Architectures<a name="architectures_usage" ></a>
The main purpose of a recommender system (RS) is to predict which item will be clicked next by the user. This is not a guarantee of positive feedback; it just counts what the user probably will click.
In order to achieve this task, we can find two major methods: Content-based and Collaborative-based. However, these methods are complex and require a deep dive into the data and the process, so our starting point has been two simple models to get familiar with RS: Random and Popularity.

If you use online streaming services to watch films, such as Prime or Netflix, professional network such as Linkedin, Youtube, Spotify or whatever, you might recognize the following expressions:
- Here are some recommendations ... [1]
- The most viewed today is ... [2]
- People with similar insterests (have seen, are following...) [3]
- Because you have seen ... [4]

These expressions respond to the method which the recommendation are based:

[1] **Random**:
This is one of the simplest methods, usually selected when new user. A list of k items is randomly selected from the whole item dataset.

![](https://hackmd.io/_uploads/H1KdpUnt2.jpg)

[2] **Popularity**:
This is the second simplest method, user thinks that most viewed items tend to be "the better". We want to be part of the group so if many others have seen I have to see it as well. 

![](https://hackmd.io/_uploads/SyJ2_PhK3.jpg)


[3] **Collaborative** Based Filtering (CF)
This model uses the opinion to recommend items identifying other users with similar taste.
This is the most mature technique and also the most common, we will see more in detail the reasons.

![](https://hackmd.io/_uploads/BkT4_v2th.png)

[4] **Content** Based Filtering (CBF)
The content-based model tries to “understand” why a user interacts with an item: because of user features (as age, genre, address, job), might be the item features (duration, theme, violence, actors…) or any other personal information.
This has a high computational cost because to deal with the large amount of features, the content is limited because it depends on the information that user provide, so normally there is overspecialization and sparsity of data.

![](https://hackmd.io/_uploads/BJnmvt6t2.png)


# Techniques explored <a name="tech_exp"></a>
The two broad approaches in recommender systems are Content-Based(CBF) and Collaborative Filtering(CF).
CF approaches have more advantatges over the amount of data required, while CBF items are vectors of features from user and items, CF just use the history of user-items interactions. The data domain of CF is scalable to any type of context.  For this reason we found more interesting go deeper into the different approaches to CF architectures:
- Matrix Factorization (MF)
- Factorization Machines (FM)
- Graph Convolutional Networks (GCN)
- Reinforcement Learning (RL)
- Deep Learning Model (DL)

Before going into a detailed analysis of these techniques we need to talk about the dataset and the preprocessing.

# Dataset <a name="dataset"></a>
Any data analysis system needs data to work with. Our dataset is downloaded from <a href="[?????](https://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a)" target=_blank>here</a>. After the initial preprocessing we have four columns of official data set used in the Netflix Prize competition. Each row contains <b>user id</b>, <b>movie id</b>, <b>rating</b> (from 1 to 5) and <b>timestamp</b> formatted as "Y/M/d" mark (1998-2005). This is basically a history register about user-movie interaction sorted by date, which is the minimum information required for CF models. However the original dataset has to be tuned before applied following many criterion which are explained in the next section.


## Preprocessing <a name="data_features"></a>
The original dataset, real data extracted from real people, has to be splitted in data to train and data to test. Also it's recommended generate a validation test to finetune the parameters if working with deep learning models to improve the results. In order to compare the results obtained in every model it's mandatory work with the same train and test dataset. 
We have considerated the next statements before splitting:
- Reduce de size of the original dataset from 100M to a subset around 1M5 of samples. (A smaller amount of data speeds up the study process because it reduces the execution time)
- Delete the samples which could generate noise. Users with less than 20 interactions and Movies with less than 5 interactions. This criterion are based on the results obtained in other studies that demonstrate good metrics.
- The timestamp mark has to be different in the same user samples. In our case, the timestamp only considered the date with no time, so if two movies are viewed the same day we couldn't know which has been the last if we have no time detail. This step is essential to split the data, because the test contains the last interacion of each user.
- As the prediction of the recommender is about the next interaction of a user-movie the rating information contained in the dataset is considered as positive sample. Whatever if it's rated with 1 or 5, we only take in consideration the fact of interaction itself. (If a user-movie is not done it's because it didn't happen yet)
- To reduce the volum of data we randomly suffle the samples and extract from them 1M5 of registers and the check that these samples meet the previous requirements.
- The indexing is an important point, the data at this moment is scattered so we will reindex the users (from 0 to ..) and the movies (from 0 to ..) adding an offset because of redundancy in future process.
- After all these steps, we will only use user and movie ids, so timestamp can be deleted as well.

So finally we obtained a dataset with 1.154.533 rows, with 7795 users and 10295 movies.

![](https://hackmd.io/_uploads/Sk6mTzy93.png=300x450) ![](https://hackmd.io/_uploads/SyVR6G1c2.png=300x450)


## Negative sampling <a name="data_neg"></a>
After the dataset records each user-item interaction as positive, there are no logs indicating non-interactions. Therefore, it becomes necessary to "manually" generate negative sampling, specifically for user-movie non-interactions. This preprocessing step is performed subsequent to the splitting process:
- Regarding to train data the negative sampling is adding 4 negative samples for 1 positive in interaction list.

![](https://hackmd.io/_uploads/B13ZpoRYn.png=200x200)

- Regarding to test data the negative sampling is adding 99 negative samples for 1 that we know it's real because it has been extracted from the original dataset. 

![](https://hackmd.io/_uploads/Sy_CyhCY3.png=120x180)

# Metrics for evaluation
The main and clear evidence that our model is working properly is the accuracy given by the Hit rate, but there are other metrics that helps us to interpret what is going on in our system.

## Hit rate <a name="hitrate"></a>
If k are the first items that our recommender outputs, hit rate is how many times the output appears in one theses k positions. The next example shows the 10 items of our GT corresponding to user 1, and as we see in the figure the first item supposed to be clicked is 14966.

![](https://hackmd.io/_uploads/Hk9uXA0th.png=150x150)

As we can see in the recommender list, the item 14966 is between the 10 first positions. So the RS is working as expected. In the context of Netflix platform this means that user will see ten movies, one of these we know user has clicked because it was in the original dataset, so the system is doing well.

![](https://hackmd.io/_uploads/ryTMX0Rtn.png)

## Intralist (NDCG) <a name="intralist"></a>
We can check how relevant the list of recommended items are, Normalized Discounted Cumulative Gain (NDCG) is a type of intralist metrics. NDCG measures the "quality" of the ranking, we assign a gain to each element of the list for example using the feedback of users (rating less than 3 gain is 0, over is 1). The items at the top of the list accumulate a higher gain than the bottom positions because of distribution of probabilities. 

![](https://hackmd.io/_uploads/S1ZP8AAFh.png)

For instance, consider the case of user 1 (up), where the NDCG score was 0.1183. This score indicates that the item they clicked was initially ranked seventh out of ten positions, while it should have been ranked among the top three positions.

## Coverage<a name="coverage"></a>
This measure assesses the breadth of recommendations given, gauging the system's capability to suggest items that cover a broad spectrum of user preferences, thus evaluating both diversity and the system's ability.

Coverage is the proportion of unique items over the total dataset that have been recommended by the system. A higher coverage indicates a greater variety of recommended items and increases the likelihood of discovering new or less known options.

The coverage metric is determined by the ratio of unique items recommended by the system to the total number of unique items in the dataset. A higher coverage value suggests a broader range of recommended items. Conversely, lower coverage may imply a system that prioritizes a subset of popular items, neglecting the diversity of user preferences.


<!--![](https://hackmd.io/_uploads/BJHj6sAth.png)-->

# CF Models Exploration:: Understanding and Comparison <a name=""></a>

## Random model:
The random model suggests 10 movies randomly to the user. Considering its nature, we anticipate poor performance from this model. Indeed, the results are as follows:

| Hit ratio 10 | NDCG     | Coverage |
| ------------ | -------- | -------- |
| 0.1001       | 0.0451     | 100%   |


## Popularity models:
We have implemented two different popularity models. The absolute popularity model returns the top 10 most popular movies in the whole dataset. If we run this model we obtain the metrics:

| Hit ratio 10 | NDCG | Coverage |
| -------- | -------- | -------- |
| 0.015     | 0.007     | 0.09% |

On the other hand, if we recommend the top 10 most popular movies from the list of possible recommendations to each user the metrics are as follows:

| Hit ratio 10 | NDCG | Coverage |
| -------- | -------- | -------- |
| 0.59     | 0.31     |   19%   |

## Matrix Factorizacion and Factorization Machine Model (FM)<a name=""></a>
Matrix factorization (MF) is a technique that consists in decompose a matrix in two smaller arrays. 
With this operation the system learns about the preferences of user and the features of the items. So we have a huge matrix of user-item that we want to separate in a matrix of users and a matrix of items. These two arrays are called *latent factor embedding* or what we know as embeddings, they contain the factors that determine the profile of a user and relates it to an item properties.
The next figure explain this process, it has been extrated from [T1].
![](https://hackmd.io/_uploads/Syvxp1kqh.png)
Simple and effective but only works with explicit data and it's not possible to add extra features as CBF.
In our dataset we use positive and negative sampling, the behaviour of a user is modelled implicitly. We need another more general model framework that can deal with the drawbacks of MF, called <b>Factorization Machines</b>. The next figure explain this process, it has been extracted from [T1].
![](https://hackmd.io/_uploads/ByPs4xk93.png)
<b>Factorization Machines (FM)</b> is an extension of Matrix Factorization that allows capturing higher order interactions between variables in a machine learning model. Unlike Matrix Factorization, Factorization Machines are capable of modeling non-linear interactions between features using factorization techniques.
Instead of working directly with latent factor matrices, they use feature combinations to represent higher-order interactions. These combinations are generated by factoring the cross products of the original features. The next equation has been implemented:
![](https://hackmd.io/_uploads/SylOweyqn.png)
FM can capture interactions at both order 2, which refers to interactions between pairs of features, and higher orders, which encompass interactions between sets of features. This capability enables the modeling of intricate and non-linear relationships between variables, thereby accommodating more complex patterns in the data.
Here we can see the results obtained in our FM model:

| Hit ratio 10 | NDCG | Coverage |
| -------- | -------- | -------- |
| 0.3374     | 0.1713   | 45,81% |

## Graph Convolutional Network Model (GCN)<a name=""></a>

As the structures become more complex, another approach can be used in combination with FM that it's called Graph Convolutional Networks (GCN). GCN are more beneficial when working in some of the following scenarios:
- Graph structure: In many cases, recommender systems may have additional information in the form of a graph that represents relationships between users, items, or other entities. For example, there may be social connections between users or similarity relationships between items. GCNs are designed to take advantage of this structure of the graph and capture richer and more contextually significant patterns and relationships.
- Contextual information: GCNs can help to integrate additional contextual information into the recommendation process. GCNs can consider contextual features such as time, location, or any other type of relevant information to improve recommendations. This ability to capture contextual information can be useful in scenarios where preferences and recommendations may vary based on context.
- Generalization to new items or users: FMs may face difficulties in generalizing to new items or users that are not present in the training data set. Instead, GCNs can take advantage of the existing graph structure and relationships to make inferences and offer recommendations even for items or users without historical data (solving cold start problem).

An illustration showcasing the functioning of a Graph Convolutional Network (GCN) is presented here. In the image, the RS model is depicted as a red node, while the yellow and blue nodes represent users and items, respectively. 
> This image has been extracted from this <a href="https://datascienceub.medium.com/2-3-recommendation-gcn-for-rs-397e98f37050">source</a>.
![](https://hackmd.io/_uploads/Hy0uplgq3.png)

You can observe that the yellow nodes and blue nodes stores relations between themselves creating complex structures and patterns that GCN can solve easily.

During the implementation process, making a simple modification is sufficient. Instead of using an embedding layer, we replace it with a GCN layer. However, in our specific scenario, the outcomes do not yield significant results due to the sparse nature of the data. As a result, these outcomes have been excluded from consideration.

## Bias Problems <a name="bias"></a>
After exploring the models above we have detected the following problems: 

-  **Cold start:** <a name="cold_start"></a>
There are two categories of cold-start:
    1. During the start-up of a recommender, the users have not interacted yet with the listed items, so there are no history to compare. 
    2. The same situation occurs when a new item is added or when a new user sings up into the system, there are no interactions. 
Due to the high number of investigations that have been developed, there are a lot of strategies to mitigate this effect. 
A common strategy when dealing with new items is to combine a collaborative filtering recommender for warm items with a content-based filtering recommender for cold items. This approach is to rely on hybrid recommenders (HR). Hybrid models can combine CBF and CF or switch between them depending on the needings.
Another option is ask to the user for explicit data and another strategy ethically questionable is integrating information from other user activities, such as browsing histories or social media platforms.
When new user or item is added, as we have seen, GCN can solve this task.

- <b>Filter bubbles: </b><a name="fil_bub"></a>
It pertains to a scenario where users of a recommender system find themselves confined within an information bubble. This occurs when they predominantly receive recommendations that closely match their preferences, which consequently restricts their exposure to a diverse range of perspectives and content.
Two major approaches are commonly used: 
    1. Incorporation of contextual information: In addition to a user's preferences, recommender systems can use additional contextual information, such as geographic location, time, current trends, or overall popularity, to broaden the diversity of recommendations. This allows you to consider the context in which the recommendation is made and offer more varied options.
    2. Explicit user feedback: Recommender systems can provide users with the option to express their preferences beyond implicit interactions. By allowing users to provide explicit feedback, such as ratings, comments, or preference specifications, you can improve personalization and reduce the undue influence of the preference bubble.
- <b>Scalability: </b><a name="scale"></a>
Scalability problems are common in recommender systems, especially when large volumes of data are handled and complex calculations must be performed in real time.
To address these scalability issues, various techniques and approaches are used, such as the use of efficient algorithms and data structures, the distribution and parallelization of computations, the use of scalable data storage and retrieval systems, and the use of data architectures.
- <b>Non-interpretable results: </b><a name="nonint"></a>
Interpretation of results in recommender systems can be difficult due to opaque algorithms, incomplete or noisy data, changing user preferences, subjective interpretation, and lack of full context. 
By applying an attention mechanism to the recommender system, it becomes possible to filter out uninformative content and select the most representative items. This approach not only enhances interpretability but also ensures that the chosen items are highly relevant.

- <b>Data hungry: </b><a name="datahung"></a>
Recommender systems based on machine learning algorithms, often need a sufficiently large and diverse data set to capture user preferences and behaviors. This is because these algorithms look for hidden patterns and relationships in the data to generate effective recommendations, more data more accurate embeddings but in the other hand more computational cost.

The model known as the *Deeper Model* operates on deeper layers within these systems, surpassing traditional ones by virtue of its enhanced processing capabilities. These models can be customized for specific tasks, effectively handling non-linear data by incorporating multiple approaches into a single system. Despite the limited data availability, we anticipate that the focus will be primarily on improving accuracy.


## Deep Models (DL)<a name="deep_models"></a>

These deeper models have an architecture with multiple hidden layers, allowing them to learn more sophisticated features and patterns:

- More abstract feature representations: With additional layers, deeper models can learn higher-level, more abstract feature representations. Each hidden layer extracts more complex and specific features compared to the previous layers. This allows for better capture of complex patterns and relationships in the data.

- Ability to learn non-linear relationships: Deeper models have a greater ability to learn non-linear relationships between input variables and output targets. As the data is propagated through multiple hidden layers, nonlinear transformations are applied repeatedly, allowing the model to capture more sophisticated and accurate nonlinear relationships.

- Long-term dependency capture: Deeper models are especially useful for capturing long-term dependencies in data. Through multiple layers, these models can learn long-term relationships and dependencies between input data, which is especially beneficial in sequential or time-sequence tasks.

- Better representation of hierarchical features: Deeper models can capture hierarchical features in the data. Each hidden layer extracts features from previous features, allowing for a richer, more hierarchical representation of the data. This is especially useful in structured data or data with complex relationships, where features depend on multiple levels of abstraction.

- Greater generalizability: Deeper models typically have greater generalizability, allowing them to better adapt to new or previously unseen data. Through multiple layers, these models can learn richer and more flexible representations, allowing them to better generalize to different scenarios and test cases.

## Proposed deep architectures
To fully explore the capabilities of deeper models, we introduce three distinct network architectures designed by us.

### Deep model:


![](https://hackmd.io/_uploads/S1-dJEW92.png)

As noted, the model consists of two embedding layers that receive input and are then concatenated. This concatenated representation is further passed through a sequence of fully connected layers. The resulting output of the network represents the probability of a match between the user and the movie.

Setting the following hyperparameters
- **batch size:** 500000
- **number of epochs:** 5
- **learning rate:** 0.025
- **embeding dimension for users:** 40
- **embeding dimension for movies:** 39

and training we obtain:

![](https://hackmd.io/_uploads/BJa1VE-c3.png)

We can see that the model begins to overfit around the 25th batch so by doing an early stoppage we get the following test metrics:

| Hit ratio 10 | NDCG | Coverage |
| -------- | -------- | -------- |
|  0.68  | 0.61     | 50%

### Residual model:

![](https://hackmd.io/_uploads/SJQeD4-ch.png)

In this case the model proposed has wider layers and a skip connection in the first layer of the fully connected block. 

Setting the following hyperparameters
- **batch size:** 500000
- **number of epochs:** 10
- **learning rate:** 0.05
- **embeding dimension for users:** 40
- **embeding dimension for movies:** 39

and training we obtain:

![](https://hackmd.io/_uploads/BJ8afBbch.png)

We can see that the model overfits faster than the previous one. In this case from the 18th batch. By doing an early stopagge we obtain the following test metrics:

| Hit ratio 10 | NDCG | Coverage |
| -------- | -------- | -------- |
|  0.52  | 0.30     | 90%        |


### Compact model:

![](https://hackmd.io/_uploads/SyjQNSW9n.png)

The compact model is the simplest of the three architectures. This model eliminates the two fully connected layers that we had after the embedings in the deep model.

Setting the following hyperparameters
- **batch size:** 500000
- **number of epochs:** 3
- **learning rate:** 0.01
- **embeding dimension for users:** 20
- **embeding dimension for movies:** 20

and training we obtain:

![](https://hackmd.io/_uploads/SkZiFBZ52.png)

| Hit ratio 10 | NDCG | Coverage |
| -------- | -------- | ------- |
|  0.55  | 0.27     | 24% |


# Results<a name="results"></a> 

|Model             | Hit ratio 10 | NDCG     | Coverage (%) |
|------            | ------------ | -------- | -------- |
|Random            | 0.10       | 0.05   | 100%     |
|ABS Popularity    | 0.02        | 0.01    | 0.09%    |
|Popularity        | 0.59         | 0.31      | 19%      |
|Factorization M   | 0.34       | 0.17   | 46%   |
|Deeper            |  0.68        | 0.61     | 50% |
|Residual          |  0.52        | 0.30     | 90% | 
|Compact           | 0.55         | 0.27     | 24% |

In the provided table, we observe that both the Hit Ratio (HR) metrics and NDCG values are similar for the random and ABS popularity models. Upon examining the user_test and recommendation list, it becomes apparent that approximately 65% of the movie_ids are greater than 10,000. Typically, an NDCG value close to 1 indicates highly relevant and well-ordered recommendations, while a value close to 0 suggests less relevant or disorganized recommendations. In other words, the recommender system is providing recommendations with a low likelihood of being useful or interesting to users, resulting in a low HR compared to other models.

When utilizing the second versions of the popularity model, the smaller setlist size increases the probability of a match, as observed. However, considering the coverage metric reveals a significantly worse result because our system only encompasses a small fraction of the entire available item base. Over time, this limited coverage can lead to a decrease in the quality of recommendations, contributing to the problem of the filter bubble.

Initially, when employing more advanced data analysis techniques such as the factorization machine model, notable improvements are observed. This is reflected in a HR of 33% and an increased NDCG score, indicating a reduction in the filter bubble issue.

Furthermore, when utilizing deep analysis models, a significant qualitative leap is witnessed. The HR increases to 68%, and the NDCG reaches 61%. All three models (Deeper, Residual, and Compact) exhibit high accuracy. However, the NDCG metric suggests that the residual and compact versions might face challenges related to the diversity of recommended items compared to the Deeper model, as their NDCG values decrease.


# Conclusions <a name="conclusion"></a>

Random and popularity recommendation systems:

- These systems are simple and easy to implement.
- They provide recommendations without considering user preferences.
- They tend to have low quality in terms of recommendation relevance.
- They do not consider personalization or diversity in recommendations.

Factorization Machine model:

- Introduces more advanced data analysis techniques.
- Improves the quality of recommendations compared to random and popularity systems.
- Enables the incorporation of user and item preferences and characteristics into the analysis.
- Increases the level of personalization and can provide more relevant recommendations.

Deep learning models:

- Represent a significant qualitative leap in terms of performance.
- Achieve a significant increase in hit rate (HR) and recommendation quality (NDCG).
- Enables learning more complex features and patterns from the data.
- They may encounter diversity issues in recommendations, especially in the compact versions.

As we transition from random and popularity-based systems to advanced models like Factorization Machine and Deep Learning, we notice enhancements in recommendation quality and personalization. Nonetheless, it's important to be mindful of potential diversity issues that can arise in these sophisticated models. The selection of a recommendation system should consider the project's specific goals, requirements, and available resources.


# References
[0] https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada

(T1) https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db 

[1] https://arxiv.org/abs/1703.04247

[2] https://arxiv.org/abs/1708.05031

[3] https://arxiv.org/abs/2103.03587

[4] https://arxiv.org/pdf/2007.09036.pdf

[5] https://datascienceub.medium.com/1-3-recommendation-vanilla-pipeline-for-recommender-systems-rs-ab7425b86d9

[6] https://datascienceub.medium.com/2-3-recommendation-gcn-for-rs-397e98f37050

[7] https://medium.com/sciforce/deep-learning-based-recommender-systems-b61a5ddd5456



