#Identifying Fraud from Enron Email
###Completed as final project for the course Intro to Machine Learning from Udacity

For the ipython notebook containing the analysis for this project visit [this link](http://willemolding.github.io/IdentifyFraudEnron/)

1. The goal of this project was to construct a classifier capable of identifying which Enron employees were deemed Persons Of Interest (POIs) in the court proceedings following the companies collapse. A person of interest is defined as a person who was either charged, settled, or testified in exchange for immunity. The ideas behind this project could be used to build systems to monitor companies for internal fraud and expose it.
The data available is comprised of two components. The first component is financial data about the employees of the company. This includes many features including salary, bonuses, and stock options. The second part of the dataset was derived from the company email records. It specifically relates to how regularly employees communicated via email with POIs and non-POIs. The intuition behind this data is that employees collectively engaged in fraud would communicate more with each other to organise fraudulent activities. 
Unfortunately both the financial and email data contained missing values. The missing fields in the financial data I concluded represented the same thing as a zero value. For example a person who does not receive a bonus would have a NaN value in the bonus field. For the financial features I replaced all the NaNs with zeros.  Some possibilities for how to deal with the missing values are:
  * Drop rows that contain missing values
  * Impute the missing values with some function of the columns
  * Develop two independent classifiers for the email and financial features and combining their outputs

	If we drop the rows containing NaNs this will mean losing almost half of our training data 	which is not acceptable. Creating two classifiers could be an elegant solution but its 	implementation may be too complex for this project. Additionally a learner would be unable 	to learn interactions between financial and email features. I decided to go with the imputation method using the median of the columns.

	Some outliers were identified in the email data. The most prominent outlier was a row containing the total of all the columns. This was likely produced in the data wrangling stage and included by error. After removing this outlier the results of the machine learning algorithms improved significantly. This highlights the importance of ensuring the data set is clean.
There was also one particular employee who send ~14000 emails which was more than double the next highest number. Presumably  this person send out mass emails possibly to the entire company on a regular bases. There were also two employees who received far more emails than any other. Since these points did not represent typical employees I decided to remove them from the data set.

2. The features I used in my final POI identifier were: 
	* salary
	* to_messages
	* deferral_payments
	* total_payments
	* exercised\_stock_options
	* bonus
	* restructed_stock
	* shared\_receipt\_with_poi
	* expenses
	* from_messages
	* other
	* deferred_income
	* long\_term_incentive
	* fraction\_of\_total\_messages\_to_poi
	* fraction\_of\_messages\_to_poi
	
	The last two features were engineered by dividing the count of messages to/from this person to a POI by the total number of messages they sent/received. This feature is as recommended  in the lecture videos for the course. Replacing the total count of messages to/from poi with the fraction yielded an increase in accuracy score of around .6% for both the decision tree classifier and the AdaBoost classifier as well as an improvement in precision and recall scores. Based on this I decided to keep the feature. 
To select the subset of features to use out of the total features set including the engineered features I used the feature importance ratings of the decision tree and AdaBoost classifiers. The feature importance ratings were sorted and graphed for both classifiers (see attached ipython notebook). Reassuringly the features were ranked very similarly by both classifiers. In both cases there were five features that were ranked much lower than the rest so these were dropped and the remaining ones made up the final feature set. 
The three classifiers that were tested (random forest, AdaBoost and naïve Bayes) are insensitive to feature scaling so no scaling was performed. 

3. The algorithm I settled on for the final submission was AdaBoost with decision trees as the base classifiers. I also evaluated random forest (bagging with decision trees) and naïve bayes. It was a interesting experiment to see how two ensemble methods that use the same base classifiers could yield very different results. 
I decided to drop the naïve Bayes classifier from the analysis before the parameter selection stage. Its performance was far below the other two classifiers and with no parameters to tune it was not going to show improvement.
After the algorithm parameter tuning stage was completed the random forest classifier yielded a higher cross validated accuracy score (85.7%) compared with the AdaBoost classsifier (84.1%). However the reason I decided to use the AdaBoost classifier for the final submission is the final scores for precision and recall ( 0.29 and 0.3) compared with the random forest (0.19 and 0.14). When the AdaBoost classifier was evaluated with the provided tester.py script it obtained better scores again (accuracy: 0.843, precision: 0.39, recall: 0.31). 

4. The parameters of machine learning algorithms can have a lot of influence on the algorithms performance on a given metric. An incorrectly tuned algorithm can give poor performance on the training set (underfitting) or poor generalization (overfitting). 
To tune the parameters of the random forest and AdaBoost classifiers I used a cross validated grid search. In both cases the parameters to be optimized related to the base decision tree classifiers. These were the splitting criterion ('gini' or 'entropy'), minimum samples per leaf node (between 1 and 50 at intervals of 5) and the max tree depth (between 1 and 10). Additionally for  AdaBoost the number of classifiers was selected (between 1 and 100 at intervals of 10). All combinations of these parameters were automatically tried by the cross validated grid search and the combination that yielded the highest cross validated score was selected. As the requirements of this assignment were to construct a classifier that exceeded 0.3 for precision and recall the metric I decided to optimize for is the f1-score. This score is an equally weighted combination of precision and recall. 

5. In this context validation is the testing of a machine learning algorithm by applying it to data that was not seen during training. The purpose of this is to get an estimate of the algorithms generalization performance. A classic mistake in validation is to not have separation between the data sets used for parameter tuning and those used for obtaining a validation score. This can lead to an over-estimate of the algorithms ability to generalize. 
I validated my analysis by using a stratified shuffle split cross validation approach. The data set is randomly split into training and test sets while preserving the proportions of true and false labels from the original data set. 100 random splits were used when selecting features an parameters and 1000 splits were used to obtain the final reported values. The mean of the scores for each split were taken as the final scores. This approach is a good way to obtain estimates given the very small data set we had access to. As a very similar approach was used for the parameter/feature selection and for obtaining our final estimates it is likely that these are over-estimates of the algorithms performance on unseen data.

6. For the final classifier the performance on the tester.py yielded
  * Accuracy: 0.843
  * Precision:  0.390
  * Recall: 0.308
  
	The accuracy score tells us that on average 84.3% of the examples in the test were classified correctly. This in itself is not a particularly impressive or informative metric. As the data set for this problem was highly unbalanced, a classifier that predicted only non-poi as its output would be able to obtain an accuracy score of 87.4%. 
The precision score is an estimate of the probability that the label of an example is true give the algorithm predicted it is true. In the context of this problem it is the probability that a person is actually a POI given our algorithm predicts they are. Randomly sampling a person from the data set the probability that they are a POI is about 0.13 while if our algorithm predicts they are a POI the precision score states that this probability jumps up to 0.39. This illustrates that we are gaining information by applying our algorithm. 
The recall score is an estimate of the probability that the algorithm will predict true for an example given the actual label is true. In our context this is the probability that the algorithm will identify someone as a POI if they in fact are. We obtained a value of 0.308 meaning that 69.2% of POIs will go undetected. In my opinion this value is too high as the cost for incorrectly labelling an innocent person as a POI is low (maybe they have to sit through a tedious court hearing) compared with the cost of labelling a true POI as false (someone may get away with fraudulently obtaining millions of dollars). As it is possible to trade off between precision and recall I would probably re-tune the algorithm to yield a better value of recall at the cost of precision for a production system.
