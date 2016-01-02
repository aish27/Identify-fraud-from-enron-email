#!/usr/bin/python
#AUTHOR: AISHWARYA VENKETESWARAN
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
 
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Testing all financial features
'''
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees']
'''
'''
('exercised_stock_options', 0.24740486725663738)
('restricted_stock', 0.17837389380530974)
('salary', 0.13148704171934261)
('total_payments', 0.11891592920353983)
('expenses', 0.096341186319062355)
('other', 0.091422566371681488)
('bonus', 0.078042825537294455)
('total_stock_value', 0.058011689787132198)
('director_fees', 0.0)
('long_term_incentive', 0.0)
('deferred_income', 0.0)
('restricted_stock_deferred', 0.0)
('loan_advances', 0.0)
('deferral_payments', 0.0)
'''
#Testing all email features
'''
features_list = ['poi','to_messages', 'from_poi_to_this_person',
 'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi'] 
'''
'''
('shared_receipt_with_poi', 0.31767905571992117)
('from_this_person_to_poi', 0.23673742240979229)
('from_messages', 0.2296674679487179)
('from_poi_to_this_person', 0.12547134238310706)
('to_messages', 0.090444711538461495)
'''
#Testing combinations
'''
features_list = ['poi','to_messages', 'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi',
'exercised_stock_options','restricted_stock','salary','total_payments', 
'expenses', 'other','bonus', 'total_stock_value'] 
'''
'''
('from_this_person_to_poi', 0.2386750804505228)
('exercised_stock_options', 0.22600000000000017)
('other', 0.19335050568900133)
('salary', 0.1167538213998391)
('expenses', 0.070052292839903385)
('bonus', 0.064452433628318442)
('shared_receipt_with_poi', 0.059457964601769907)
('restricted_stock', 0.031257901390644785)
('total_stock_value', 0.0)
('total_payments', 0.0)
('from_messages', 0.0)
('from_poi_to_this_person', 0.0)
('to_messages', 0.0)
'''
#I think logically, to_messages is a feature that must be important
#Hence, I am going to continue to keep it in the feature list.
'''
features_list = ['poi', 'from_this_person_to_poi',  'shared_receipt_with_poi',
'exercised_stock_options','restricted_stock','salary', 'expenses', 'other',
            'bonus','to_messages' ] 
'''
'''
('other', 0.24129172917444439)
('bonus', 0.19831880651773121)
('expenses', 0.16861679861679865)
('exercised_stock_options', 0.16784274193548393)
('from_this_person_to_poi', 0.14050274657836476)
('to_messages', 0.083427177177177167)
('salary', 0.0)
('restricted_stock', 0.0)
('shared_receipt_with_poi', 0.0)
'''

'''
features_list = ['poi', 'from_this_person_to_poi', 
'exercised_stock_options','expenses', 'other','bonus','to_messages' ] 
'''
'''
('exercised_stock_options', 0.26423667120341454)
('bonus', 0.26308699423665005)
('other', 0.23984484715449525)
('from_this_person_to_poi', 0.10900468272171245)
('expenses', 0.096948937711250585)
('to_messages', 0.026877866972477064)
'''
'''
features_list = ['poi', 'from_this_person_to_poi', 
'exercised_stock_options','expenses', 'other','bonus' ]
''' 
'''
('expenses', 0.31489963512256097)
('exercised_stock_options', 0.27207185020655639)
('bonus', 0.26344038469090025)
('other', 0.1055009379663885)
('from_this_person_to_poi', 0.044087192013593866)
'''
'''
features_list = ['poi', 'exercised_stock_options','expenses', 'other','bonus']
'''
'''
('expenses', 0.31487729325551977)
('bonus', 0.28996456930040648)
('other', 0.25383494212584623)
('exercised_stock_options', 0.14132319531822751)
'''
'''
features_list = ['poi','expenses', 'other','bonus']
'''
'''
('expenses', 0.45246405251161115)
('other', 0.2909755967737897)
('bonus', 0.25656035071459921)
'''

#features_list = ['poi','expenses', 'other']
#Accuracy: 0.79827	Precision: 0.44924	Recall: 0.48450


#Using 'other' and 'expenses' features provide the best values considering all 
#the features I have tested. 
#features_list = ['poi','expenses', 'other']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
my_dataset_temp= {}
for i,j in zip(data_dict.keys(),data_dict.values()):
  if i!='TOTAL':
    my_dataset_temp[i] = j
    
### Task 3: Create new feature(s)
# I have created three new features:
# fraction_from_poi: A fraction between 0 and 1 representing what fraction of
# messages were from a poi out of all the messages. 
# fraction_to_poi:  A fraction between 0 and 1 representing what fraction of 
#messages were addressed to a poi out of all the messages
# fraction_all_poi: A fraction between 0 and 1 representing the extent of 
#interactions of a poi and a person out of all the messages (considering the ‘from’ and ‘to’ fields). 

    
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if all_messages !='NaN' and all_messages!=0.0:
          fraction = poi_messages/(1.0* all_messages)
    return fraction

my_dataset= my_dataset_temp
for i,data_point in zip(my_dataset_temp.keys(),my_dataset_temp.values()):
        
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person,
                                            to_messages )

        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi,from_messages)
        #print i, fraction_from_poi,fraction_to_poi
        
        my_dataset[i]["fraction_from_poi"] = fraction_from_poi
        my_dataset[i]["fraction_to_poi"] = fraction_to_poi
        my_dataset[i]["fraction_all_poi"] = (fraction_to_poi*0.5) + (fraction_from_poi*0.5)


'''
features_list = ['poi','expenses', 'other', "fraction_to_poi",
                 "fraction_from_poi","fraction_all_poi"]
'''
'''
('expenses', 0.38048383891600113)
('other', 0.26402791049530183)
('fraction_to_poi', 0.22110834478021996)
('fraction_from_poi', 0.073139063317634714)
('fraction_all_poi', 0.061240842490842488)
'''
'''
features_list = ['poi','expenses', 'other', "fraction_to_poi",
                 "fraction_from_poi"]
'''
'''
('other', 0.3620132584806498)
('expenses', 0.35073828684902053)
('fraction_to_poi', 0.22110834478021996)
('fraction_from_poi', 0.066140109890109849)
'''
'''
features_list = ['poi','expenses', 'other',"fraction_to_poi"]
'''
'''
('other', 0.44598482709648707)
('expenses', 0.29008375682288751)
('fraction_to_poi', 0.26393141608062543)
Accuracy: 0.79608	Precision: 0.39049	Recall: 0.39850
'''

#features_list = ['poi','expenses', 'other']
#Accuracy: 0.79827	Precision: 0.44924	Recall: 0.48450

#Using 'other' and 'expenses' features provide the best values considering all 
#the features I have tested. 
features_list = ['poi','expenses', 'other']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a variety of classifiers

#Classifier #1
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

'''
Got a divide by zero when trying out: GaussianNB()
Precision or recall may be undefined due to a lack of true positive predicitons
Total time taken 2.60199999809
'''

#Classifier #2
#from sklearn.svm import SVC
#clf = SVC()
'''
Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, 
class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
Precision or recall may be undefined due to a lack of true positive predicitons
Total time taken 4.53500008583
'''

#Classifier #3
#from sklearn.ensemble import RandomForestClassifier
#clf  = RandomForestClassifier()
#Accuracy: 0.80473	Precision: 0.43236	Recall: 0.23650

#Classifier #4
#clf = DecisionTreeClassifier()
#Accuracy: 0.80000	Precision: 0.45396	Recall: 0.49300

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

### Tuning the algorithm

##Trying various values for max_depth:
#clf =  DecisionTreeClassifier(max_depth=3)
#Accuracy: 0.74182	Precision: 0.17290	Recall: 0.11100
#clf =  DecisionTreeClassifier(max_depth=7)
#Accuracy: 0.80709	Precision: 0.46741	Recall: 0.43750
#clf =  DecisionTreeClassifier(max_depth=9)
#Accuracy: 0.80227	Precision: 0.45871	Recall: 0.48600
#clf =  DecisionTreeClassifier(max_depth=12)
#Accuracy: 0.80055	Precision: 0.45488	Recall: 0.48900
#clf =  DecisionTreeClassifier(max_depth=8)
#Accuracy: 0.80700	Precision: 0.47019	Recall: 0.48500
#max_depth=8 provides the best performance.
 
##Trying various values for max_features:
#clf = DecisionTreeClassifier(max_features="sqrt")
#Accuracy: 0.77400	Precision: 0.38655	Recall: 0.41400
#clf = DecisionTreeClassifier(max_features="log2")
#Accuracy: 0.78182	Precision: 0.40449	Recall: 0.42350
#clf = DecisionTreeClassifier(max_features="auto")
#Accuracy: 0.77600	Precision: 0.39098	Recall: 0.41600
#clf = DecisionTreeClassifier()
#Accuracy: 0.80000	Precision: 0.45396	Recall: 0.49300
#Using None results in best performance.
 
##Trying differnt values for random_state:
#clf = DecisionTreeClassifier(random_state=42)
#Accuracy: 0.80309	Precision: 0.46114	Recall: 0.49250
#clf = DecisionTreeClassifier(random_state=49)
#Accuracy: 0.79918	Precision: 0.45178	Recall: 0.48950
#clf = DecisionTreeClassifier(random_state=60)
#Accuracy: 0.79827	Precision: 0.44924	Recall: 0.48450
#clf = DecisionTreeClassifier(random_state=44)
#Accuracy: 0.79845	Precision: 0.44965	Recall: 0.48450
#clf = DecisionTreeClassifier(random_state=41)
#Accuracy: 0.79991	Precision: 0.45345	Recall: 0.48950
#clf = DecisionTreeClassifier()
#Accuracy: 0.80000	Precision: 0.45396	Recall: 0.49300
#Using random_state = 42 gives the best performance. 

#Final tuned classifier:
clf = DecisionTreeClassifier(random_state=42,max_depth=8)
#Accuracy: 0.80736	Precision: 0.47085	Recall: 0.48050
'''
('other', 0.75985212626115917)
('expenses', 0.24014787373884086)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best')
	Accuracy: 0.80736	Precision: 0.47085	Recall: 0.48050	
       F1: 0.47562	F2: 0.47854
	Total predictions: 11000	True positives:  961    False positives: 1080
	False negatives: 1039	True negatives: 7920

Total time taken 1.8789999485
'''

#Validation and Evaluation
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
features_train = transformer.fit_transform(features_train).toarray()
features_test = transformer.fit_transform(features_test)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
