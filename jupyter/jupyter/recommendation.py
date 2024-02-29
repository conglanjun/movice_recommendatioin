import json
import math

import numpy as np
import pandas as pd

class Recommend:
    def __init__(self, needInit=False, n=100) -> None:
        self.n = n
        self.ratings = pd.read_csv('ml-25m/ratings.csv', index_col=None)
        self.movies = pd.read_csv('ml-25m/movies.csv', index_col=None)
        self.user_list = self.ratings['userId'].drop_duplicates().values.tolist()[:self.n]
        self.movie_list = self.movies['movieId'].drop_duplicates().values.tolist()
        self.genres_list = self.movies['genres'].values.tolist()
        self.type_list = self.get_type_list()

        self.type_map, self.type_map_reverse = self.get_list_index_map(self.type_list)
        self.user_map, self.user_map_reverse = self.get_list_index_map(self.user_list)
        self.movie_map, self.movie_map_reverse = self.get_list_index_map(self.movie_list)
        self.ratings_matrix = self.get_rating_matrix()
        # self.user_favor_matrix = self.get_user_favor_matrix()
        if needInit:
            self.user_sim_matrix = self.get_user_sim_matrix(self.ratings_matrix)
            np.savetxt('user_sim_matrix.csv', self.user_sim_matrix, delimiter = ',')
        else: 
            self.user_sim_matrix = np.loadtxt(open("./user_sim_matrix.csv", "rb"), delimiter=",")

    def get_list_index_map(self, list):
        """
        list to map
        """
        map = {}
        map_reverse = {}
        for i in range(len(list)):
            map[list[i]] = i
            map_reverse[i] = list[i]
        return map, map_reverse

    def get_type_list(self):
        """
        get type list
        """
        type_list = []
        for item in self.genres_list:
            movie_types = item.split('|')
            for movie_type in movie_types:
                if movie_type not in type_list and movie_type != '(no genres listed)':
                    type_list.append(movie_type)
        return type_list

    def get_rating_matrix(self):
        """
        construct rating matrix
        """
        matrix = np.zeros((len(self.user_map.keys()), len(self.movie_map.keys())))
        for row in self.ratings.itertuples(index=True, name='Pandas'):
            userIdNum = getattr(row, "userId")
            if userIdNum > 100:
                break
            user = self.user_map[userIdNum]
            movie = self.movie_map[getattr(row, "movieId")]
            rate = getattr(row, "rating")
            matrix[user, movie] = rate
        print(matrix)
        return matrix

    def get_user_favor_matrix(self):
        """
        user favorite
        """
        matrix = np.zeros((len(self.user_list), len(self.type_list)))
        for user in range(len(self.user_list)):
            print(user)
            weight = 0
            rating = self.ratings_matrix[user]
            for movie in range(len(rating)):
                if rating[movie] != 0:
                    types = self.genres_list[movie].split('|')
                    for t in types:
                        if t in self.type_map.keys():
                            matrix[user][self.type_map[t]] += rating[movie]
                            weight += rating[movie]
            matrix[user] /= weight
        return matrix

    def cosine_similarity(self, list1, list2):
        """
        consine
        """
        res = 0
        d1 = 0
        d2 = 0
        for index in range(len(list1)):
            val1 = list1[index]
            val2 = list2[index]
            res += val1 * val2
            d1 += val1 ** 2
            d2 += val2 ** 2
        return res / (math.sqrt(d1 * d2))


    def get_user_sim_matrix(self, input_matrix):
        """"
        construct similarity matrix
        """
        size = len(input_matrix)
        matrix = np.zeros((size, size))
        for i in range(size):
            print(f'sim i: {i}')
            for j in range(i + 1, size):
                sim = self.cosine_similarity(input_matrix[i], input_matrix[j])
                matrix[i, j] = sim
                matrix[j, i] = sim
        return matrix

    def k_neighbor(self, matrix, index, k):
        line = matrix[index]
        tmp = []
        for i in range(len(line)):
            tmp.append([i, line[i]])
        tmp.sort(key=lambda val:val[1], reverse=True)
        return tmp[:k]

    def get_predict(self, matrix, index, k):
        neighbors = self.k_neighbor(matrix=matrix, index=index, k=k)
        all_sim = 0
        rate = [0 for i in range(len(self.ratings_matrix[0]))]
        for pair in neighbors:
            neighbor_index = pair[0]
            neighbor_sim = pair[1]
            all_sim += neighbor_sim
            rate += self.ratings_matrix[neighbor_index] * neighbor_sim
        rate /= all_sim
        return rate

    def get_CFRRecommend(self, matrix, index, k, n):
        rate = self.get_predict(matrix, index, k)
        for i in range(len(rate)):
            if self.ratings_matrix[index][i] != 0:
                rate[i] = 0
        res = []
        for i in range(len(rate)):
            res.append([i, rate[i]])
        res.sort(key=lambda val:val[1], reverse=True)
        return res[:n]
    
    def recommend(self, userId):
        recommendDict = {}
        for otherUserId in range(len(self.user_list)):
            if otherUserId == userId:
                continue
            sim = self.cosine_similarity(self.ratings_matrix[userId], self.ratings_matrix[otherUserId])
            otherRating = self.ratings_matrix[otherUserId]
            for otherMovieId in range(len(otherRating)):
                if self.ratings_matrix[userId][otherMovieId] == 0:
                    recommendScore = otherRating[otherMovieId] * sim
                    recommendDict[otherMovieId] = recommendScore
        return sorted(recommendDict.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)

