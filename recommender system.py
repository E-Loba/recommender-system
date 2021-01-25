"""
Lobaskin Egor
6930-30-2170
Graduate School of Informatics
Intelligence Science and Technology, Speech and Audio Processing
Pattern recognition assignment 1

data is a courtesy of MovieLens and GroupLens
https://grouplens.org/datasets/movielens/latest/
"""

import csv
from matplotlib import pyplot
from numpy import cumsum
from math import sqrt, ceil

# construct a database for each user's movie ratings as a nested map
# {user: {movie: rating, ...}, ...}
# {movie: {user: rating, ...}, ...}
user_ratings = {}
movie_ratings = {}
rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
total_items = 0
with open("ratings.csv", "r") as ratings:
    print("constructing user-movie-ratings map")
    ratingsreader = csv.reader(ratings)
    for row in ratingsreader:
        try:
            parsed_rating = float(row[2])
        except ValueError:
            continue
        if row[0] in user_ratings:
            user_ratings[row[0]][row[1]] = parsed_rating
        else:
            user_ratings[row[0]] = {row[1]: parsed_rating}
        if row[1] in movie_ratings:
            movie_ratings[row[1]][row[0]] = parsed_rating
        else:
            movie_ratings[row[1]] = {row[0]: parsed_rating}
        try:
            rating_key = ceil(parsed_rating)
            rating_counts[rating_key] = rating_counts[rating_key] + 1
        except ValueError as E:
            pass
total_items = sum(rating_counts.values())
print(rating_counts)
print(total_items)

"""
Bayesian (naive) approach:
P(rating | user, movie) =
P(u, m | r)P(r) / [P(u,m|r)P(r) + P(u,m|r')P(r')] =
P(r|u)P(r|m)P(r') / [P(r|u)P(r|m)P(r') + P(r'|u)P(r'|m)P(r)]
"""


def cond_prob(item, condition):
    """
    the conditional probability P(r|c) of a rating (integer) given a user or movie (in map format) as condition
    applies parsing to the values in the map in order to compare, rounds upwards
    if item is negative, computes the compliment of the probability P(r'|c)
    :param item: the rating as an integer
    :param condition: user or movie as map of {string: string}
    :return: P(r|c) = |instances of r| / |total instances in c|
    """
    abs_item = abs(item)
    counter = 0
    for entry in condition:
        if abs_item == ceil(condition[entry]):
            counter = counter + 1
    if item > 0:
        return counter / len(condition)
    else:
        return (len(condition) - counter) / len(condition)


def naive_prediction(user_id, movie_id):
    """
    uses the bayesian formula to make a naive prediction:
    P(rating | user, movie) =
    P(u, m | r)P(r) / [P(u,m|r)P(r) + P(u,m|r')P(r')] =
    P(r|u)P(r|m)P(r') / [P(r|u)P(r|m)P(r') + P(r'|u)P(r'|m)P(r)]
    this approach is naive, because it assumes that both the user distribution and the movie distribution
    can be modelled independently, and that simply joining the distributions gives an accurate result
    :param user_id: the id of a user. the data is fetched from the user ratings map
    :param movie_id: the id of a movie, the data is fetched from the movie ratings map
    :return: the rating with the highest probability, conditioned on the user and movie distributions
    """
    user_dict = user_ratings[user_id]
    movie = movie_ratings[movie_id]
    previous = 0
    prediction = 0
    for number in range(1, 6):
        prior_complement = (total_items - rating_counts[number]) / total_items
        cpr_u = cond_prob(number, user_dict)
        cpr_m = cond_prob(number, movie)
        try:
            prob = (cpr_u * cpr_m * prior_complement) / \
                ((cpr_u * cpr_m * prior_complement) + ((1 - cpr_u) * (1 - cpr_m) * (1 - prior_complement)))
        except ZeroDivisionError:
            # for rating 'r', there might be 0 instances of a user giving, or a movie receiving that particular rating
            prob = 0
        if prob > previous:
            prediction = number
            previous = prob
        else:
            continue
    return prediction


def z_score(mydict):
    """
    converts the entries in the dictionary into z scores
    z score = (i - mean) / standard_deviation
    :param mydict: dictionary in the format {string: string} where the values are convertible to floating point numbers
    :return: a dummy dictionary with the same keys, but z scores for values
    """
    length = len(mydict)
    avg = sum(mydict[rating] for rating in mydict) / length
    std = sqrt(sum((mydict[rating] - avg) ** 2 for rating in mydict) / (length - 1))
    return dict([(entry, (mydict[entry] - avg)/std) for entry in mydict])


def pearson_weight(dict1, dict2):
    """
    compute the pearson coefficient between two dictionaries.
    returns a "weight" value between 0 and 1 that can be used for weighted sums
    calculates the z scores for each entry in order to compute the pearson coefficient
    :param dict1: content 1
    :param dict2: content 2
    :return: (pearson coefficient + 1) / 2
    """
    intersection = set(dict1) & set(dict2)
    if len(intersection) < 2:
        return 0.5
    else:
        z_scores1 = z_score(dict1)
        z_scores2 = z_score(dict2)
        return ((sum((z_scores1[item] * z_scores2[item]) for item in intersection)/(len(intersection)-1)) + 0.99)/2


def regression_weight(dict1, dict2):
    """
    computes a squared distance between two dictionaries as the average of squared differences of the common items
    :param dict1: content 1
    :param dict2: content 2
    :return: sqrt( sum((i1 - i2)**2) / N ) where N is the size of intersection of the dictionaries
    """
    intersection = set(dict1) & set(dict2)
    if len(intersection) < 2:
        return 0.5
    else:
        return sqrt(sum((dict1[item] - dict2[item])**2
                        for item in intersection) / len(intersection))


def advanced_prediction(user_id, movie_id, weight_method):
    """
    uses the given weight method to compute a prediction
    prediction = sum( weight * user rating ) / number of users
    where the weight =  given method
    and user rating = rating given to a user by the movie
    thus, the prediction for a user is the weighted average of the ratings given by all other users
    :param user_id: this id is used to fetch data from the user ratings map
    :param movie_id: this id is used to feth data from the movie ratings map
    :param weight_method: the method used to compute weights. can be pearson weight or regression weight
    :return: an average of other users' ratings, weighted by the given method
    """
    relevant_users = list(movie_ratings[movie_id].keys())
    weights = dict([(x, weight_method(user_ratings[user_id], user_ratings[x])) for x in relevant_users if x != user_id])
    if len(weights) < 1:
        # some movies don't have enough data to compute a prediction. in that case, apply the user's average rating
        return sum(user_ratings[user_id][rating] for rating in user_ratings[user_id]) / len(user_ratings[user_id])
    else:
        return sum([user_ratings[x][movie_id] * weights[x] for x in relevant_users if x != user_id])\
               / sum(list(weights.values()))


"""
calculate the error of prediction
"""
print("calculating average squared loss")
per_user_squared_naive_loss = []
per_user_squared_pearson_loss = []
per_user_squared_regression_loss = []
user_count = 0
# with open("results.txt", "w") as myfile:
for usr in user_ratings:
    user_count = user_count + 1
    per_user_squared_naive_loss.append(0)
    per_user_squared_pearson_loss.append(0)
    per_user_squared_regression_loss.append(0)
    movie_count = 0
    for mov in user_ratings[usr]:
        movie_count = movie_count + 1
        actual = user_ratings[usr][mov]
        naive = naive_prediction(usr, mov)
        per_user_squared_naive_loss[-1] = per_user_squared_naive_loss[-1] + (naive - actual)**2
        adv = advanced_prediction(usr, mov, pearson_weight)
        reg = advanced_prediction(usr, mov, regression_weight)
        per_user_squared_pearson_loss[-1] = per_user_squared_pearson_loss[-1] + (adv - actual)**2
        per_user_squared_regression_loss[-1] = per_user_squared_regression_loss[-1] + (reg - actual)**2
    per_user_squared_naive_loss[-1] = per_user_squared_naive_loss[-1] / movie_count
    per_user_squared_pearson_loss[-1] = per_user_squared_pearson_loss[-1] / movie_count
    per_user_squared_regression_loss[-1] = per_user_squared_regression_loss[-1] / movie_count
    # myfile.write("\nuser: " + usr + " movie: " + mov +
    #              " actual rating: %.2f\t || naive: %.2f\tregression: %.2f\tpearson: %.2f"
    #              %(actual, naive, reg, adv))
    print("\nuser: %3d || naive loss: %.2f\tregression loss: %.2f\tpearson loss: %.2f"
          % (int(usr), per_user_squared_naive_loss[-1], per_user_squared_regression_loss[-1],
             per_user_squared_pearson_loss[-1]))
print("average squared loss of naive prediction: ")
print(sum(per_user_squared_naive_loss) / user_count)
print("average squared loss of regressive prediction: ")
print(sum(per_user_squared_regression_loss) / user_count)
print("average squared loss of pearson prediction: ")
print(sum(per_user_squared_pearson_loss) / user_count)

cumulative_naive = cumsum(per_user_squared_naive_loss)
cumulative_regression = cumsum(per_user_squared_regression_loss)
cumulative_pearson = cumsum(per_user_squared_pearson_loss)
pyplot.plot(cumulative_naive, c="blue")
pyplot.plot(cumulative_regression, c="red")
pyplot.plot(cumulative_pearson, c="green")
pyplot.show()
