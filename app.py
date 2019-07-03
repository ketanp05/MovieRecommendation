'''
Recommendation Systems: the ML algorithm will learn our likes and
recommend what option would be best for us. These learning algorithms
are getting accurate as time passes
Types:
1)Collaborative Systems: predict what you like based on other similar
users have liked in the past
2)Content-Based: predict what you like based on what you have liked
in the past
eg:Netflix combines both approaches to predict your likes more accurately

APP: this script reads in a dataset of movie ratings and recommends new
movies for users

Dependencies: numpy, scipy, lightfm
lightfm: helps in performing bunch of recommendation algos,a great lib
to start with for building recommendation systems
'''
import numpy as np

#lets use the 'fetch_movielens' method from submodule datasets
#try diff methods to obtain diff results and compare the accuracy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch the dataset and format it
#we will be using MovieLens dataset(available on Kaggle)
data = fetch_movielens(min_rating=4.0)

#make interaction matrix from the csv and store it in data as a dictionary
print(repr(data['train']))
print(repr(data['test']))

#loss means loss func which measures loss = (model pred - desired output)
#we minimize it during the training to gain more accuracy
#Weighted Approx Rank Pairwise-warp
model = LightFM(loss='warp')

#epochs-no of runs, num_threads = parallel computation
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    #no of users and movies using shape attribute of dicts
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        #csr - compressed sparse row format
        #tocsr is a subarray which will retrieve using the indices attributes
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model will predict
        scores = model.predict(user_id, np.arange(n_items))

        #sort them in order of their scores
        #scores in desc order because of the negative sign
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out user_ids
        print("Users %s" % user_id)
        print("     Known Positives:")

        #top 3 known_positives the user has picked
        for x in known_positives[:3]:
            print("     %s" % x)

        #top 3 recommended movies predicted by our model
        print("     Recommended:")
        for x in top_items[:3]:
            print("     %s" %x)

'''
print('Enter 3 random ids:')
idList = []
for i in range(3):
    idList = int(input('ENTER:'))
'''

#enter in 3 random userids
sample_recommendation(model, data, [4, 45, 89])

'''
def main():
    input('Enter 3 random user ids:')
    idList = []
    for i in range(3):
        idList = int(input('ENTER:'))
    sample_recommendation(model, data, idList)

if __name__ == '__main__':
    main()
'''