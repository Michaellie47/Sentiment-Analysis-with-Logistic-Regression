import math
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

sns.set()
%matplotlib inline

# We will use a dataset consisting of food product reviews on Amazon.com source.

products = pd.read_csv('food_products.csv')

# Set seed for the whole program
np.random.seed(416)

products.head()

# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment. Let's see how many of each rating we have.
products = products[products['rating'] != 3].copy()
len(products)

plt.title('Number of reviews with a given rating')
sns.histplot(products['rating'])

# Now, we will assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or lower are negative. 
# For the sentiment column, we use +1 for the positive class label and -1 for the negative class label.

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products.head()

# Now, we can see that the dataset contains an extra column called sentiment which is either positive (+1) or negative (-1).

# We want to remove punctuations using built-in function
def remove_punctuation(text):
    """
    Remove any punctuation in text. Python has a default set of 
    punctuation marks, stored in string.punctuation, that contains
    !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    """
    if type(text) is str:
        return text.translate(str.maketrans('', '', string.punctuation))
    else:
        return ''
    
products['review_clean'] = products['review'].apply(remove_punctuation)

# Next, we use scikit-learn's CountVectorizer to get counts for each word.
# Make counts
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(products['review_clean'])

# Get the feature names (e.g., one per word)
features = vectorizer.get_feature_names_out()

# Make a new DataFrame with the counts information
product_data = pd.DataFrame(count_matrix.toarray(),
        index=products.index,
        columns=features)

# Add the old columns to our new DataFrame. 
# We won't use review_clean and the summary in our model, but we will keep them to look at later.
product_data['sentiment'] = products['sentiment']
product_data['review_clean'] = products['review_clean']  
product_data['summary'] = products['summary']

product_data.head()

# Now we split the data into training, validation, and test sets (with random state = 3)
train_data, test_and_validation_data = train_test_split(product_data, test_size=0.2, random_state=3)
validation_data, test_data = train_test_split(test_and_validation_data, test_size=0.5, random_state=3)

# Next, we want to compute both the most frequent label; and the validationaccuracy of the majority class classfier (store as a number between 0 and 1)
# True values (1), False values (0)

# Compute the occurences within the sentiment column and return the maximum count in the train set.
majority_label_train = train_data['sentiment'].value_counts().idxmax()

# Compute the classifier accuracy = true values (sentiment validation = majority_label_train) / total validation_data
majority_classifier_accuracy = (validation_data['sentiment'] == majority_label_train).sum() / len(validation_data)

majority_label = majority_label_train
majority_classifier_validation_accuracy = majority_classifier_accuracy

# Now we would want to train a sentiment classifier with Logistic Regression (using scikit-learn's)
sentiment_model = LogisticRegression(penalty='l2', C=1e23, random_state=1)
sentiment_model.fit(train_data[features], train_data['sentiment'])

# Let's look at some of the coefficients and the corresponding words. 
# The weights are stored in a coef_ property.
coefficients = sentiment_model.coef_

print(sentiment_model.coef_)
print('Smallest coefficient', coefficients.min())
print('Largest coefficient:', coefficients.max())
print(sentiment_model.coef_[0])

# Next, using the sentiment model we trained above, compute the word with the most negative weight and the word with the most positive weight.

# Get the feature names (e.g., one per word)
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

# Store the coefficients in the sentiment_model into coefficinents
coefficients = sentiment_model.coef_[0]

# Initialize a dictionary to and map the feature_names with coeffs stored
feature_coeffs = dict(zip(feature_names, coefficients))

# Sorting the values (coeffs) in the feature_coeffs dictionary from highest to lowest
highest_lowest_coeffs = sorted(feature_coeffs.items(), key = lambda x: abs(x[1]), reverse=True)

# Printing the values within the dictionary from highest to lowest (top 5)
for word, coeff in highest_lowest_coeffs[:5]:
    print(f"{word}: {coeff}")

# Compute most negative weight word
most_negative_word = min(feature_coeffs, key = feature_coeffs.get)

# Compute most positive weight word
most_positive_word = max(feature_coeffs, key = feature_coeffs.get)

print('Most negative word:', most_negative_word)
print('Most positive word:', most_positive_word)

# Now, that the model is trained, we can make predictions on the validation data.
# First, we will restrict to only at 3 examples in the validation dataset (sample_data).
sample_data = validation_data[8:11]
sample_data[['sentiment', 'review_clean', 'summary']]

# Start by predicting the probability of positive/negative sentiment of the 3 examples in the sample_data.
print('  Prob Negative, Prob Positive')
print(sentiment_model.predict_proba(sample_data[features]))

# We are also able to predictions labels using the predict function.
print('Predicted labels')
print(sentiment_model.predict(sample_data[features]))


# Now, let's make predictions on the validation data without any restrictions.

# Find the index of the most positive review in the validation set
index_most_pos = np.argmax(sentiment_model.predict_proba(validation_data[features])[:, 1])

# Find the index of the most negative review in the validation set
index_most_neg = np.argmax(sentiment_model.predict_proba(validation_data[features])[:, 0])

# Applying the Hint below

# using .iloc[] to get the row and find review_clean value

most_positive_review = validation_data['review_clean'].iloc[index_most_pos]
most_negative_review = validation_data['review_clean'].iloc[index_most_neg]

print('Most Positive Review:')
print(most_positive_review)
print()
print('Most Negative Review:')
print(most_negative_review)

# We notice that both reviews are pretty long, then we can normalize the counts to avoid prioritizing long reviews over shorter ones (do this on next project)

# Now, we compute the validation accuracy for the sentiment model.

# Get the predicted_labels 
predicted_labels = sentiment_model.predict(validation_data[features])

# Initialize actual values sentiment in validation split
validation_sentiment = validation_data['sentiment']

# Compute the accuracy using accuracy_score(y_true, y_pred)
sentiment_model_validation_accuracy = accuracy_score(validation_sentiment, predicted_labels)

# Generate a confusion matrix to analyze the performance of the predictor.
def plot_confusion_matrix(tp, fp, fn, tn):
    """
    Plots a confusion matrix using the values 
       tp - True Positive
       fp - False Positive
       fn - False Negative
       tn - True Negative
    """
    data = np.matrix([[tp, fp], [fn, tn]])

    sns.heatmap(data,annot=True,xticklabels=['Actual Pos', 'Actual Neg']
              ,yticklabels=['Pred. Pos', 'Pred. Neg']) 
    

# Plotting the confusion matrix to show the number of True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).
# initializations of tp, fp, tn, fn

tp = 0
fp = 0
tn = 0
fn = 0


# For Loop to plot the confusion matrix
for predicted, output in zip(predicted_labels, validation_sentiment):
    if predicted == 1 and output == 1: # For True Positive
        tp += 1
    elif predicted == 1 and output == -1: # For False Positive
        fp += 1
    elif predicted == -1 and output == 1: # For False Negative
        fn += 1
    elif predicted == -1 and output == -1: # For True Negative
        tn += 1


# To help avoid overfitting, we apply l2 regularization to our Logistic Regression model.

# Set up the regularization penalities to try
l2_penalties = [0.01, 1, 4, 10, 1e2, 1e3, 1e5]
l2_penalty_names = [f'coefficients [L2={l2_penalty:.0e}]' 
                    for l2_penalty in l2_penalties]

# Q6: Add the coefficients to this coef_table for each model
coef_table = pd.DataFrame(columns=['word'] + l2_penalty_names)
coef_table['word'] = features

# Q7: Set up an empty list to store the accuracies (will convert to DataFrame after loop)
accuracy_data = []

for l2_penalty, l2_penalty_column_name in zip(l2_penalties, l2_penalty_names):
    # TODO(Q6 and Q7): Train the model. Remember to pass `fit_intercept=False` and `random_state=1` to the model.
    
    model = LogisticRegression(penalty = 'l2', C = 1/l2_penalty, fit_intercept = False, random_state = 1)
    model.fit(train_data[features], train_data['sentiment'])

    # TODO(Q6): Save the coefficients in coef_table

    coef_table[l2_penalty_column_name] = model.coef_[0]

    # TODO(Q7): Calculate and save the train and validation accuracies

    train_accuracy = model.score(train_data[features], train_data['sentiment'])
    val_accuracy = model.score(validation_data[features], validation_data['sentiment'])

    # Same approach as in HW 1, to build up a list of dictionaries, and then convert that to a DataFrame with the values described.
    accuracy_data.append({
        'l2_penalty': l2_penalty,
        'train_accuracy': train_accuracy,
        'validation_accuracy': val_accuracy,
    })


    
accuracies_table = pd.DataFrame(accuracy_data)


# Look at coef_table
coef_table

# Look at accuracies_table
accuracies_table

# Lastly, we inspect the coefficients and plot the effect of increasing the l2 penalty on the 10 words we select below.
positive_words = coef_table[['word', 'coefficients [L2=1e+00]']].nlargest(5, 'coefficients [L2=1e+00]')['word']
negative_words = coef_table[['word', 'coefficients [L2=1e+00]']].nsmallest(5, 'coefficients [L2=1e+00]')['word']

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    """
    Makes a plot of coefficients, given a table where rows correspond to words and
    columns correspond to the l2 penalty, a list of positive words, a list of 
    negative words, and a list of l2 penalties.
    """
    def get_cmap_value(cmap, i, total_words):
        """
        Computes a nice scaling of from i=0 to i=total_words - 1
        for the given cmap
        """
        return cmap(0.8 * ((i + 1) / (total_words * 1.2) + 0.15))


    def plot_coeffs_for_words(ax, words, cmap):
        """
        Given an axes to plot on and a list of words and a cmap,
        plots the coefficient paths for each word in words
        """
        words_df = table[table['word'].isin(words)]
        words_df = words_df.reset_index(drop=True)  # To make indices sequential

        for i, row in words_df.iterrows():
            color = get_cmap_value(cmap, i, len(words))
            ax.plot(xx, row[row.index != 'word'], '-',
                    label=row['word'], linewidth=4.0, color=color)

    # Make a canvas to draw on
    fig, ax = plt.subplots(1, figsize=(10, 6))
   
    # Set up the xs to plot and draw a line for y=0
    xx = l2_penalty_list
    ax.plot(xx, [0.] * len(xx), '--', linewidth=1, color='k')

    # Plot the positive and negative coefficient paths
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    plot_coeffs_for_words(ax, positive_words, cmap_positive)
    plot_coeffs_for_words(ax, negative_words, cmap_negative)

    # Set up axis labels, scale, and legend  
    ax.legend(loc='best', ncol=2, prop={'size':16}, columnspacing=0.5 )
    ax.set_title('Coefficient path')
    ax.set_xlabel('L2 penalty ($\lambda$)')
    ax.set_ylabel('Coefficient value')
    ax.set_xscale('log')


make_coefficient_plot(coef_table, positive_words, negative_words, l2_penalty_list=l2_penalties)


