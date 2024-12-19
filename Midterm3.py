#Emmalyn Holmquist
#12/08/2024
#CPSC 507

#------------------------------------

#Problem 1

#Create a function called occurs_n_times which will take as input a list L, and a positive integer n
# and will return True if there is an element in L which occurs n or more times and False otherwise.
# In this problem, you may not use imports and you may not use dictionaries.
# You may use any available operation on lists though.
# Your function must run efficiently and will be tested on lists with many elements.

#define function
def occurs_n_times(list, n):
    #initialize a dictionary to store new values and counts
    dict = {}
    #for loop to go through each element in list
    for value in list:
        #if the element is not in the list, add it to dict with count 1
       if value not in dict:
           dict[value] = 1
        #if the element is in the dict, increment the count by 1
       elif value in dict:
           dict[value] += 1
        #check if the incremented count is at n yet. if it is, we found an element including at least n times. return true
       if dict[value] == n:
            return True
    #if we make it through the whole list without finding n times elements, then return false
    return False

#Test
# list = ["hello", "bye", 3, 4, 33, 5, "hello", "hello", 3, "hello", "hello"]
# if_occurs= occurs_n_times(list, 6)
# print(if_occurs)
# if_occurs= occurs_n_times(list, 3)
# print(if_occurs)

#---------------------------------------

#Problem 2

#In order to reduce my email load, I decide to implement a machine learning algorithm to
# decide whether or not I should read an email, or simply file it away instead.
# To train my model, I obtain the following data set of binary-valued features about each email,
# including whether I know the author or not, whether the email is long or short, and whether it
# has any of several key words, along with my final decision about whether to read it
# (y = +1 for ”read,” y = −1 for ”discard”).

# Import necessary libraries
import numpy as np

# Define a function to calculate distances between data points in Xtrain and Xtest
def get_distances(Xtrain, Xtest):
    # Calculate the dot product of each row in Xtrain and reshape it to a column vector
    dot_products_train = (Xtrain ** 2).sum(axis=1).reshape(Xtrain.shape[0], 1)
    # Calculate the dot product of each row in Xtest and reshape it to a row vector
    dot_products_test = (Xtest ** 2).sum(axis=1).reshape(1, Xtest.shape[0])
    # Calculate the dot product between Xtrain and the transpose of Xtest
    dot_products_train_test = np.dot(Xtrain, Xtest.T)
    # Compute the pairwise distances between the data points and return the result
    return (dot_products_train - 2 * dot_products_train_test + dot_products_test).T
def find_most_frequent_label(nearest_labels):
    d = {}
    for label in nearest_labels:
        if label not in d:
            d[label] = 1
        else:
            d[label] += 1
    max_freq_label = None
    max_feq = 0
    for label in d.keys():
        if d[label] > max_feq:
            max_feq = d[label]
            max_freq_label = label
    return max_freq_label


#initialize training data as a list
email_train_list = [[0, 0, 1, 1, 0], [1, 1, 0, 1, 0], [0, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0], [1, 0, 1, 1, 1], [0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0], [1, 0, 1, 1, 0], [1, 1, 1, 1, 1]]
#convert list to numpy array
email_train = np.array(email_train_list)

#training data labels
email_train_choices = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1]
email_train_labels = np.array(email_train_choices).T

#testing points
email_test_list = [[0,0,0,0,0], [1, 1, 0, 1, 0]]
email_test = np.array(email_test_list)

# Calculate distances between training and test data using the defined function
distances = get_distances(email_train, email_test)


##a - Compute the class prediction for x = (0,0,0,0,0) using a k-NN classifier with k = 3.
# What about for x = (1,1,0,1,0) using k = 3? (10 points)

######
# 3NN code:
#######

print("\n3NN test:")

print("\nPredictions for each point:\n")

for row in distances:
    closest_points_3 = np.argpartition(row, 3)[:3]
    predictions_3 = email_train_labels[closest_points_3]
    # print(predictions_2)
    prediction = find_most_frequent_label(predictions_3)
    print(" max freq label is: ", prediction)

##b - Compute the class prediction for x = (0,0,0,0,0) using a k-NN classifier with k = 5.
# What about for x = (1,1,0,1,0) using k = 5? (10 points)

######
# 5NN code:
#######

print("\n5NN test:")

print("\nPredictions for each point:\n")

for row in distances:
    closest_points_5 = np.argpartition(row, 5)[:5]
    predictions_5 = email_train_labels[closest_points_5]
    # print(predictions_2)
    prediction = find_most_frequent_label(predictions_5)
    print(" max freq label is: ", prediction)

##c - For each of the above predictions, explain how the classifier works, including the concept of k-NN and how it
# determines the class label. (10 points)

#For each of the above, the first max freq label output is the class predictor for the first point (0,0,0,0,0) and the
#second max freq output is the class predictor for the second point (1,1,0,1,0). To determine the predicted class labels, the
# classifier works by finding the first k closest data points/neighbors from the training data set. In particular, for k = 3,
#we will compute the distance between the new point and each point in the training set, and select the three data points with
#the smallest distance from the new/test data point. In determining the class of the new data, we then look at what the majority class
#is among the closest data points of the training set. This is why we have a seperate np array for the training data labels,
#as we first find the 3 closest data points from the np array with only the point features and then look at their classes.
#For k = 5, we perform a similar test, although we compute distances to find the closest 5 points in the data set. This is why it
#is possible for us to get two different classes, because we expanded our search of potential classes when we looked at
#the closest 5 points compared to the closest 3 points.

#The code works by first extracting/setting up the data in np arrays for easier computation. We have a seperate np array for labels
#than point features, as mentioned. We also have a seperate np array for our test points. We then pass our train and test points to the get
#distances function to obtain the distance between each point. The agrgmin function then allows us to determine the closest points based
#on how many we are looking for. Since we are looking for multiple points, we put this in a for loop to find the first 3 closest. To find the
#predictions of these, we then call the find most frequent label to match labels to the points.

##d - Discuss the choice of the value of k in k-NN. How might changing the value of k affect the predictions, and why? (10 points)

#For odd vs. even k, it is smartest to choose an odd k because we might get a discrepancy if there are even amounts of different classes.
#Further, at a lower value of k, the class prediction is much more sensitive to its neighbors as compared to a higher value of k, where
# in turn the prediction is based on a more global perspective and may miss finer details of the data. However, lower values of k may risk
#higher variance, and false predictions if the training set contains outliers. Thus, we must find a balance between minimal variance and
#minimal global influence.

#---------------------------------------

#Problem 3

# In this problem, you will use the file docword.nytimes.txt.gz, which is available here: Bag of Words.
# You can read more about this file at the link.
# Note that this file is compressed and will need to be extracted.
# Make sure to do this well in advance of the deadline in case of difficulties.
# When developing solutions for this problem, you may want to work with a smaller portion of the dataset.

#load data sets and store in variables
nytimes = open("docword.nytimes.txt", "r")

# # (5 points) a: Which document has the most total words? Store this value in a variable called most words.
#
#define a function to find the highest count of a document within a given text collection
def highest_count(file):
    #initialize the previous doc ID
    prev_doc = 1
    #intialize the word count
    word_count = 0
    #initialize the count of the highest amount
    highest_num_count = 0
    #initialize doc id
    doc_ID = 1
    #initialize a dictionary for storing word counts (for use in a later part)
    dict_word_counts = {}
    #go through each line of the file
    for line in file:
        #split the line into a list
        curr_line = line.split()

        # this condition is necessary because our first 3 lines of each file will only contain one num
        # and contain info about the data overall, not about a certain doc
        if len(curr_line) == 3:

            #if the doc id is the same as the previous doc id
            if int(curr_line[0]) == prev_doc:
                #incrememnt the word count
                word_count += int(curr_line[2])

            #if the doc id does not equal the previous doc id
            elif int(curr_line[0]) != prev_doc:
                #we have reached the end of the work count for that doc. If the highest count is less than the
                #currently obtained count, replace it

                if highest_num_count < word_count:
                    #set the highest num count as the current word count
                    highest_num_count = word_count
                    # set doc_id to be the doc id of the doc with highest word count
                    doc_ID = prev_doc

                #add to dictionary
                dict_word_counts[prev_doc] = word_count
                #reset word count to 1 (because we are starting at first word of next doc)
                word_count = 1
                #increment the prev doc (we move on to the next doc)
                prev_doc = int(curr_line[0])

    #we need to account for last doc in data set now. see if its a higher word count
    if highest_num_count < word_count:
        highest_num_count = word_count
        doc_ID = prev_doc

    #add to dictionary
    dict_word_counts[prev_doc] = word_count
    #return the highest doc name and the highest num words once we have gone through whole file
    return doc_ID, highest_num_count, dict_word_counts

highest_doc_nytimes, most_words, dict_word_counts=highest_count(nytimes)
print("\n")
print("The document with the most total words has ID ",highest_doc_nytimes," and has ",most_words," words.")

nytimes.close()


# (5 points) b: Which document has the most unique words? Store this value in a variable called most unique words.

#load data sets and store in variables
nytimes = open("docword.nytimes.txt", "r")


#sums the amount of keys, used for unique words dict
def total_dict(dict):
    sum_keys = 0
    for key in dict:
        sum_keys += 1
    return sum_keys


def highest_unique(file):
    #initialize the previous doc ID
    prev_doc = 1
    #initialize the count of the highest amount
    highest_num_unique = 0
    #initialize doc id
    doc_ID_uniques = 1
    #initialize dict for num unique words of all docs
    dict_docs_uniques={}
    #initialize dict for unique words of current doc
    dict_curr_uniques={}

    #go through each line of the file
    for line in file:
        #split the line into a list
        curr_line = line.split()

        # this condition is necessary because our first 3 lines of each file will only contain one num
        # and contain info about the data overall, not about a certain doc
        if len(curr_line) == 3:
            #if the doc id is the same as the previous doc id
            if int(curr_line[0]) == prev_doc:

                #check if the current word is in our current dictionary
                if int(curr_line[1]) not in dict_curr_uniques and int(curr_line[2]) == 1:
                    #if its not, add it. we don't need to do anything to it if its already in it
                    dict_curr_uniques[int(curr_line[1])] = 1

            #if the doc id does not equal the previous doc id- we are at a new one
            elif int(curr_line[0]) != prev_doc:
                #find the total amount of uniques in current dict
                curr_uniques = total_dict(dict_curr_uniques)
                #see if the amount of uniques is higher than our current value
                if highest_num_unique < curr_uniques:
                    highest_num_unique = curr_uniques
                    doc_ID_uniques= prev_doc
                #add the current uniques totals to place in dict that analyzes total uniques in each doc (for use later)
                dict_docs_uniques[prev_doc] = curr_uniques
                #adjust the prev_doc variable to be the doc of the current line (new doc)
                prev_doc = int(curr_line[0])
                #reset the dictionary of current words of document to contain nothing
                dict_curr_uniques = {}
                #add the current line into current doc dict (because the line we are currently at is the first line of new doc)
                dict_curr_uniques[int(curr_line[1])] = 1

    #account for last doc in data set now
    # find the total amount of uniques in current dict
    curr_uniques = total_dict(dict_curr_uniques)
    if highest_num_unique < curr_uniques:
        highest_num_unique = curr_uniques
        doc_ID_uniques= prev_doc
    # add the current uniques totals to place in dict that analyzes total uniques in each doc (for use later)
    dict_docs_uniques[prev_doc] = curr_uniques

    #return the highest doc name and the highest num words once we have gone through whole file
    return doc_ID_uniques, highest_num_unique, dict_docs_uniques

doc_ID_uniques, most_unique_words, dict_docs_uniques = highest_unique(nytimes)
print("The document with the most unique words has ID ",doc_ID_uniques," with ",most_unique_words," unique words.")

nytimes.close()


# (10 points) c: We will define the lexical richness of a document to be the total number of distinct words in a
# document divided by the total number of words in the document.
# What is the average lexical richness of the documents? Store this in the variable average_lexical_richness.

#Notes: lexical richness = unique words/total words. average lexical richness = sum of lexical richness/total lexical richness
# -- note that in the previous two parts, we simultaneously had a dictionary storing the document number as the key and total words
#as the value (dict_word_counts), and another with doc ID as key and unique words as value (dict_docs_uniques). we will use these in our calculations

#initialize a dict to hold lexical richness of each doc
dict_lexical_richness = {}

#we can use the spand of the dict_word_counts dict for all of the documents
for document_id in dict_word_counts:
    #add each lexical richness to the dict_lexical_richness by accessing the uniques and counts dicts
    dict_lexical_richness[document_id] = dict_docs_uniques[document_id]/dict_word_counts[document_id]
#initialize variable to hold total lexical richness, sum of all lexical richness across docs
total_richness = 0
for document_id in dict_lexical_richness:
    total_richness += dict_lexical_richness[document_id]

#find the amount of docs we are averaging
total_words = total_dict(dict_lexical_richness)

#find average
average_lexical_richness = total_richness/total_words
print("Average lexical richness across docs: ", average_lexical_richness)

# (10 points) d: Of all the documents that have at least 50 unique words, which one has the greatest lexical richness?
# Store this in a variable called greatest_lexical_richness.

#define function that will find and return the greatest lexical richness for docs with over 50 unique words
def greatest_richness_50(dict_unique, dict_lex_rich):

    #intialize variable for storing greatest lexical richness
    greatest = 0
    greatest_doc_id = 1
    for document_id in dict_unique:
        #check if there is at least 50 unique words
        if int(dict_unique[document_id]) >= 50:
            #if the lexical richness for respective doc is greater than the current greatest, become the current greatest
            if greatest < (dict_lex_rich[document_id]):
                greatest = (dict_lex_rich[document_id])
                greatest_doc_id = document_id


    return greatest, greatest_doc_id

greatest_lexical_richness, greatest_doc_id = greatest_richness_50(dict_docs_uniques, dict_lexical_richness)
print("The greatest lexical richness of docs with more than 50 unique words is ", greatest_lexical_richness, "in doc ", greatest_doc_id)
