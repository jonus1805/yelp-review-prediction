
#########################################Importing the packages##########################


import json

import pandas as pd

import numpy as np

from collections import Counter

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 

#######################################Reading the business.json file#######################

data = []

#Reading the hotel names

with open('C:\\Users\\Jonus\\Desktop\\stevens\\2nd Sem\\BIA 660B\\yelp_dataset_challenge_round9\\yelp_dataset_challenge_round9~\\yelp_academic_dataset_business.json', 'r',encoding='utf-8') as f:

    for line in f:

        data.append(json.loads(line))

business=data 

business_id=[]

business_name=[]

business_attributes=[]

business_categories=[]



for i in business:

    business_id.append(i['business_id'])

    business_name.append(i['name'])

    business_attributes.append(i['attributes'])

    business_categories.append(i['categories'])

    

#creating a data frame

busines=[]

busines.append(business_id)

busines.append(business_name)

busines.append(business_attributes)

busines.append(business_categories)

busines_df=pd.DataFrame(busines)

busines_df=busines_df.transpose()

busines_cols=['business_id','name','attributes','categories']

busines_df.columns=busines_cols

#busines_df.to_csv('C:\\Users\\Jonus\\Desktop\\stevens\\2nd Sem\\BIA 660B\\yelp_dataset_challenge_round9\\business.csv',index=False)


#########eliminating business id with othercategories[Except restaurant]################################

busines_df.replace('', np.nan, inplace=True)

busines_df.dropna( inplace=True)

busines_df['Res']=['Restaurants' in row for row in busines_df['categories']]

busines_df_final = busines_df[busines_df['Res'] == True]

#busines_df_final.to_csv('C:\\Users\\Jonus\\Desktop\\stevens\\2nd Sem\\BIA 660B\\yelp_dataset_challenge_round9\\fbusiness.csv',index=False)

############################################Reading the review.json file###########################

#Reading the user names

data1 = []

#Reading the hotel names

with open('C:\\Users\\Jonus\\Desktop\\stevens\\2nd Sem\\BIA 660B\\yelp_dataset_challenge_round9\\yelp_dataset_challenge_round9~\\yelp_academic_dataset_review.json', 'r',encoding='utf-8') as k:

    for line in k:

        data1.append(json.loads(line))

review=data1      

review_user_id=[]

review_business_id=[]

review_stars=[]



for i in review:

    review_user_id.append(i['user_id'])

    review_business_id.append(i['business_id'])

    review_stars.append(i['stars'])



#creating a data frame

reviews=[]

reviews.append(review_user_id)

reviews.append(review_business_id)

reviews.append(review_stars)

reviews_df=pd.DataFrame(reviews)

reviews_df=reviews_df.transpose()

reviews_cols=['user_id','business_id','stars']

reviews_df.columns=reviews_cols

    

#reviews_df.to_csv('C:\\Spring 2017\\Web Analytics\\Yelp Data Set\\reviews.csv',index=False)



#############getting the business id for the user###################################

user_with_business=reviews_df.sort_values(by=['user_id'])

#user_with_business.to_csv('C:\\Spring 2017\\Web Analytics\\Yelp Data Set\\user_business.csv',index=False)



#############################merging two data frames on business_id######################

merge_df = pd.merge(user_with_business, busines_df_final, on='business_id')

#result = pd.concat([user_with_business, busines_df_final], on='business_id', join='inner')

#merge_df.to_csv('C:\\Users\\Jonus\\Desktop\\stevens\\2nd Sem\\BIA 660B\\yelp_dataset_challenge_round9\\merge_user.csv',index=False)

merge_df.columns
#################################features#########################################

Accepts_Credit_Cards=[]

kids=[]

bike_parking=[]

Alcohol=[]

Romantic=[]

Intimate=[]

Classy=[]

Hipster=[]

Divey=[]

Touristy=[]

Trendy=[]

Upscale=[]

Casual=[]

caters=[]

Drive_Thru=[]

Has_TV=[]

Noise_Level=[]

Outdoor_Seating=[]

Delivery=[]

Takes_Reservations=[]

Table_service=[]

Take_out=[] 

Wi_Fi=[]

Good_For_Groups=[]

Wheelchair_Accessible=[]

like_user=[]

dessert=[]

latenight=[]

lunch=[]

dinner=[]

breakfast=[]

brunch=[]

#like_business=[]

user_id=list(merge_df['user_id'])

#business_id=[merge_df['business_id']]



for i in merge_df['attributes']:

    if 'BusinessAcceptsCreditCards: True' in i:

        Accepts_Credit_Cards.append(1)

    else:

        Accepts_Credit_Cards.append(0)

for i in merge_df['attributes']:

    if 'GoodForKids: True' in i:

        kids.append(1)

    else:

         kids.append(0)

for i in merge_df['attributes']:

    if 'BikeParking: False' in i:

        bike_parking.append(0)

    else:

        bike_parking.append(1)

for i in merge_df['attributes']:

    if 'Alcohol: none' in i:

        Alcohol.append(0)

    else:

        Alcohol.append(1)


for i in merge_df['attributes']:

    if 'Caters: False' in i:

        caters.append(0)

    else:

        caters.append(1)

for i in merge_df['attributes']:

    if 'DriveThru: False' in i:

        Drive_Thru.append(0)

    else:

        Drive_Thru.append(1)

for i in merge_df['attributes']:

    if 'HasTV: False' in i:

        Has_TV.append(0)

    else:

        Has_TV.append(1)

for i in merge_df['attributes']:

    if 'NoiseLevel: quiet' in i:

        Noise_Level.append(0)

    else:

        Noise_Level.append(1)

for i in merge_df['attributes']:

    if 'OutdoorSeating: False' in i:

        Outdoor_Seating.append(0)

    else:

        Outdoor_Seating.append(1)

for i in merge_df['attributes']:

    if 'RestaurantsDelivery: False' in i:

        Delivery.append(0)

    else:

        Delivery.append(1)

for i in merge_df['attributes']:

    if 'RestaurantsReservations: False' in i:

        Takes_Reservations.append(0)

    else:

        Takes_Reservations.append(1)

for i in merge_df['attributes']:

    if 'RestaurantsTableService: False' in i:

        Table_service.append(0)

    else:

        Table_service.append(1)

for i in merge_df['attributes']:

    if 'RestaurantsTakeOut: True' in i:

        Take_out.append(1)

    else:

        Take_out.append(0)

for i in merge_df['attributes']:

    if 'WiFi: no' in i:

        Wi_Fi.append(1)

    else:

        Wi_Fi.append(0)

for i in merge_df['attributes']:

    if 'RestaurantsGoodForGroups: False' in i:

        Good_For_Groups.append(0)

    else:

        Good_For_Groups.append(1)

for i in merge_df['attributes']:

    if 'WheelchairAccessible: False' in i:

        Wheelchair_Accessible.append(0)

    else:

        Wheelchair_Accessible.append(1) 



for i in merge_df['attributes']:

    if 'romantic: True' in i:

        Romantic.append(1)

    else:

        Romantic.append(0)      

for i in merge_df['attributes']:

    if 'intimate: True' in i:

        Intimate.append(1)

    else:

        Intimate.append(0)       

for i in merge_df['attributes']:

    if 'classy: True' in i:

        Classy.append(1)

    else:

        Classy.append(0)

for i in merge_df['attributes']:

    if 'hipster: True' in i:

        Hipster.append(1)

    else:

        Hipster.append(0) 

for i in merge_df['attributes']:

    if 'divey: True' in i:

        Divey.append(1)

    else:

        Divey.append(0)

for i in merge_df['attributes']:

    if 'touristy: True' in i:

        Touristy.append(1)

    else:

        Touristy.append(0)        

for i in merge_df['attributes']:

    if 'trendy: True' in i:

        Trendy.append(1)

    else:

        Trendy.append(0) 

for i in merge_df['attributes']:

    if 'upscale: True' in i:

        Upscale.append(1)

    else:

        Upscale.append(0)    

for i in merge_df['attributes']:

    if 'casual: True' in i:

        Casual.append(1)

    else:

        Casual.append(0)   

for i in merge_df['attributes']:

    if 'dessert:True' in i:

        dessert.append(1)

    else:

        dessert.append(0)        

for i in merge_df['attributes']:

    if 'latenight:True' in i:

        latenight.append(1)

    else:

        latenight.append(0)

for i in merge_df['attributes']:

    if 'lunch:True' in i:

        lunch.append(1)

    else:

        lunch.append(0)         

for i in merge_df['attributes']:

    if 'dinner:True' in i:

        dinner.append(1)

    else:

        dinner.append(0)

for i in merge_df['attributes']:

    if 'breakfast:True' in i:

        breakfast.append(1)

    else:

        breakfast.append(0)

for i in merge_df['attributes']:

    if 'brunch:True' in i:

        brunch.append(1)

    else:

        brunch.append(0)           

for i in merge_df['stars_x']:

    if i<=2:

        like_user.append(0)

    elif i>=3 :

        like_user.append(1)

    else:

        like_user.append('NA')

        

##################################creating a dataset with features###################################

model=[]

#model.append(busines_df_final['business_id'])

model.append(user_id)

#model.append(business_id)

model.append(Accepts_Credit_Cards)

model.append(kids)

model.append(bike_parking)

model.append(Alcohol)

#model.append(romantic)

model.append(caters)

model.append(Drive_Thru)

model.append(Has_TV)

model.append(Noise_Level)

model.append(Outdoor_Seating)

model.append(Delivery)

model.append(Takes_Reservations)

model.append(Table_service)

model.append(Take_out)

model.append(Wi_Fi)

model.append(Good_For_Groups)

model.append(Wheelchair_Accessible)

model.append(Romantic)

model.append(Intimate)

model.append(Classy)

model.append(Hipster)

model.append(Divey)

model.append(Touristy)

model.append(Trendy)

model.append(Upscale)

model.append(dessert)

model.append(latenight)

model.append(lunch)

model.append(dinner)

model.append(breakfast)

model.append(brunch)

model.append(like_user)



model_df=pd.DataFrame(model) 

model_df=model_df.transpose()  

model_cols=['user_id','Accepts_Credit_Cards','kids','bike_parking','Alcohol','caters','Drive_Thru','Has_TV','Noise_Level','Outdoor_Seating','Delivery','Takes_Reservations','Table_service','Take_out','Wi_Fi','Good_For_Groups','Wheelchair_Accessible','Romantic','Intimate','Classy','Hipster','Divey','Touristy','Trendy','Upscale','dessert','latenight','lunch','dinner','breakfast','brunch','like_user']    

model_df.columns=model_cols





###################################3removing like = NA################################33

model_df.replace('NA', np.nan, inplace=True)

model_df.dropna( inplace=True)



#model_df.to_csv('C:\\Spring 2017\\Web Analytics\\Yelp Data Set\\model.csv',index=False)
################################Converting the features into category#########################

model_df['user_id']=model_df['user_id'].astype('category')

model_df['Accepts_Credit_Cards']=model_df['Accepts_Credit_Cards'].astype('category')

model_df['kids']=model_df['kids'].astype('category')

model_df['bike_parking']=model_df['bike_parking'].astype('category')

model_df['Alcohol']=model_df['Alcohol'].astype('category')

model_df['caters']=model_df['caters'].astype('category')

model_df['Drive_Thru']=model_df['Drive_Thru'].astype('category')

model_df['Has_TV']=model_df['Has_TV'].astype('category')

model_df['Noise_Level']=model_df['Noise_Level'].astype('category')

model_df['Outdoor_Seating']=model_df['Outdoor_Seating'].astype('category')

model_df['Delivery']=model_df['Delivery'].astype('category')

model_df['Takes_Reservations']=model_df['Takes_Reservations'].astype('category')

model_df['Table_service']=model_df['Table_service'].astype('category')

model_df['Take_out']=model_df['Take_out'].astype('category')

model_df['Wi_Fi']=model_df['Wi_Fi'].astype('category')

model_df['Good_For_Groups']=model_df['Good_For_Groups'].astype('category')

model_df['Wheelchair_Accessible']=model_df['Wheelchair_Accessible'].astype('category')

model_df['Romantic']=model_df['Romantic'].astype('category')

model_df['Intimate']=model_df['Intimate'].astype('category')

model_df['Classy']=model_df['Classy'].astype('category')

model_df['Hipster']=model_df['Hipster'].astype('category')

model_df['Divey']=model_df['Divey'].astype('category')

model_df['Touristy']=model_df['Touristy'].astype('category')

model_df['Trendy']=model_df['Trendy'].astype('category')

model_df['Upscale']=model_df['Upscale'].astype('category')

model_df['dessert']=model_df['dessert'].astype('category')

model_df['latenight']=model_df['latenight'].astype('category')

model_df['lunch']=model_df['lunch'].astype('category')

model_df['dinner']=model_df['dinner'].astype('category')

model_df['breakfast']=model_df['breakfast'].astype('category')

model_df['brunch']=model_df['brunch'].astype('category')

model_df['like_user']=model_df['like_user'].astype('category')


#######################################finding the user with most reviews#################
from collections import Counter

classfi_df_user=model_df.sort_values(by=['user_id'])

dic= Counter(classfi_df_user.user_id)


c=dic.most_common()[1:350]
#('bLbSNkLggFnqwNNzzq-Ijw', 1118),
#('d_TBs6J3twMy9GChqUEXkg', 830),


classfi_df_user=model_df.sort_values(['user_id'])

user_rank1= classfi_df_user[classfi_df_user['user_id'] == 'bLbSNkLggFnqwNNzzq-Ijw'] #0.842261904762

user_rank2= classfi_df_user[classfi_df_user['user_id'] == 'd_TBs6J3twMy9GChqUEXkg'] #0.843373493976

user_rank3= classfi_df_user[classfi_df_user['user_id'] == 'cMEtAiW60I5wE_vLfTxoJQ'] #0.983050847458


#splitting into training and test dataset

X=user_rank1.iloc[:,1:30]

y=user_rank1.iloc[:,31]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#Random Forest
from sklearn.ensemble import RandomForestClassifier



RF_classifier=RandomForestClassifier(n_estimators=5)


RF_classifier.fit(X_train, y_train)

preds = RF_classifier.predict(X_test)


from sklearn.metrics import accuracy_score

print (accuracy_score(preds,y_test))

print(classifier.feature_importances_)


#####################################Feature IMportance###################################
classifier=RandomForestClassifier(n_estimators=5)

 

classifier.fit(X_train, y_train)
importances = classifier.feature_importances_
#len(importances)

std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


"""
Top Ten Important Features:

Take out
bike_parking
Alcohol
Noise_level
'Outdoor_Seating'
'Takes_Reservations'
'Table_service'
kids
'Good_For_Groups'
Caters

"""
"""
#Logistic Regression
from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression()

LR_classifier.fit(X_train, y_train)

preds = LR_classifier.predict(X_test)

#Linear SVC
from sklearn.svm import LinearSVC
SVC_classifier = LinearSVC()

SVC_classifier.fit(X_train, y_train)

preds = SVC_classifier.predict(X_test)


#NB
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


RF_classifier=RandomForestClassifier(random_state=30)
KNN_classifier=KNeighborsClassifier()
LREG_classifier=LogisticRegression()


predictors=[('knn',KNN_classifier),('lreg',LREG_classifier),('rf',RF_classifier)]

VT=VotingClassifier(predictors)


#=======================================================================================
#build the parameter grid
KNN_grid = [{'n_neighbors': [1,3,5,7,9,11,13,15,17], 'weights':['uniform','distance']}]

#build a grid search to find the best parameters
gridsearchKNN = GridSearchCV(KNN_classifier, KNN_grid, cv=5)

#run the grid search
gridsearchKNN.fit(X_train.as_matrix(),list(y_train))

#gridsearchKNN.fit(X_train.as_matrix(),list(y_train))

#y_train.astype(object).columns
              
#=======================================================================================

#build the parameter grid
#DT_grid = [{'max_depth': [3,4,5,6,7,8,9,10,11,12],'criterion':['gini','entropy']}]

RF_grid = { "n_estimators"      : [250, 300],
           "criterion"         : ["gini", "entropy"],
           "max_features"      : [3, 5],
           "max_depth"         : [10, 20],
           "min_samples_split" : [2, 4] ,
           "bootstrap": [True, False]}

#build a grid search to find the best parameters
gridsearchRF  = GridSearchCV(RF_classifier, RF_grid, cv=5)

#run the grid search
gridsearchRF.fit(X_train.as_matrix(),list(y_train))

#=======================================================================================

#build the parameter grid
LREG_grid = [ {'C':[0.5,1,1.5,2],'penalty':['l1','l2']}]

#build a grid search to find the best parameters
gridsearchLREG  = GridSearchCV(LREG_classifier, LREG_grid, cv=5)

#run the grid search
gridsearchLREG.fit(X_train.as_matrix(),list(y_train))

#=======================================================================================

VT.fit(X_train.as_matrix(),list(y_train))
 
#use the VT classifier to predict
predicted=VT.predict(X_test.as_matrix())

#print the accuracy
from sklearn.metrics import accuracy_score
print (accuracy_score(predicted,list(y_test)))

#confusion Matrix
"""
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(predicted,y_test)
"""






         

    

