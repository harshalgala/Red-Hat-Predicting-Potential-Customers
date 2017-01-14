
#Loading the libraries
library(MASS)
library(leaps)
library(randomForest)
library(e1071)
library(rpart)

#Reading the dataset
setwd("~/Desktop/Udit/CSC522/Project/Data")
#act_test=read.csv('act_test.csv')
act_train=read.csv('act_train.csv')
people=read.csv('people.csv')

#Preprocessing training set
data_train = merge(people,act_train,by="people_id")
print(names(data_train))
lapply(data_train,class)
print(str(data_train))

#Remove redundant features in training set
data_train$activity_id=NULL
data_train$date.x=NULL
data_train$date.y=NULL
data_train$people_id=NULL
#data_train$group_1=NULL


#Partition training data based on type of activity
data_train_type1=subset(data_train, data_train$activity_category=="type 1")
data_train_type1$char_10.y=NULL
data_train_type1$activity_category=NULL
data_train_rest=subset(data_train, data_train$activity_category!="type 1")
data_train_rest$char_1.y=NULL
data_train_rest$char_2.y=NULL
data_train_rest$char_3.y=NULL
data_train_rest$char_4.y=NULL
data_train_rest$char_5.y=NULL
data_train_rest$char_6.y=NULL
data_train_rest$char_7.y=NULL
data_train_rest$char_8.y=NULL
data_train_rest$char_9.y=NULL
data_train_type1=droplevels(data_train_type1)
data_train_rest=droplevels(data_train_rest)


#Test of independence for type 1 activities
outcome_type1=data_train_type1$outcome
char_38_type1=data_train_type1$char_38
data_train_type1$outcome=NULL
data_train_type1$char_38=NULL
n_type1=nrow(data_train_type1)
remove_type1=c()
for(i in 1:(ncol(data_train_type1)-1)){
  for(j in (i+1):ncol(data_train_type1)){
    tbl=table(data_train_type1[,i], data_train_type1[,j])
    c=chisq.test(tbl)
    v=c$statistic/(n_type1*(min(nrow(tbl), ncol(tbl))-1))
    if(v > 0.7){
      remove_type1=c(remove_type1, i)
      print(i)
      break
    }

  }
}
data_train_type1=data_train_type1[, -remove_type1]
data_train_type1$outcome=outcome_type1
data_train_type1$char_38=char_38_type1


#Test of independence for rest of the activities
outcome_rest=data_train_rest$outcome
char_38_rest=data_train_rest$char_38
data_train_rest$outcome=NULL
data_train_rest$char_38=NULL
n_rest=nrow(data_train_rest)
remove_rest=c()
#remove_rest=c(1,2,4,7,9,22,23,37,39)
for(i in 1:(ncol(data_train_rest)-1)){
  for(j in (i+1):ncol(data_train_rest)){
    tbl=table(data_train_rest[,i], data_train_rest[,j])
    c=chisq.test(tbl)
    v=c$statistic/(n_rest*(min(nrow(tbl), ncol(tbl))-1))
    if(v > 0.7){
      remove_rest=c(remove_rest, i)
      print(i)
      break
    }

  }
}
data_train_rest=data_train_rest[, -remove_rest]
data_train_rest$outcome=outcome_rest
data_train_rest$char_38=char_38_rest

#Supervised ratio for type1
positive=subset(data_train_type1, data_train_type1$outcome==1)
negative=subset(data_train_type1, data_train_type1$outcome==0)
p=rep(0,length(unique(data_train_type1$char_1.y)))
names(p)=unique(data_train_type1$char_1.y)
for(i in names(p)){
  p[i]=nrow(subset(positive, positive$char_1.y==i))
}
n=rep(0,length(unique(data_train_type1$char_1.y)))
names(n)=unique(data_train_type1$char_1.y)
for(i in names(n)){
  n[i]=nrow(subset(negative, negative$char_1.y==i))
}
sr=rep(0,length(unique(data_train_type1$char_1.y)))
names(sr)=unique(data_train_type1$char_1.y)
for(i in names(sr)){
  sr[i]=p[i]/(p[i]+n[i])
}
data_train_type1$char_1.ynew=rep(nrow(data_train_type1), 0)
for(i in 1:nrow(data_train_type1)){
  data_train_type1$char_1.ynew[i]=sr[as.character(data_train_type1$char_1.y[i])]
}
data_train_type1$char_1.y=NULL

#Supervised ratio for rest
positive=subset(data_train_rest, data_train_rest$outcome==1)
negative=subset(data_train_rest, data_train_rest$outcome==0)
p=rep(0,length(unique(data_train_rest$char_10.y)))
names(p)=unique(data_train_rest$char_10.y)
for(i in names(p)){
  p[i]=nrow(subset(positive, positive$char_10.y==i))
}
n=rep(0,length(unique(data_train_rest$char_10.y)))
names(n)=unique(data_train_rest$char_10.y)
for(i in names(n)){
  n[i]=nrow(subset(negative, negative$char_10.y==i))
}
sr=rep(0,length(unique(data_train_rest$char_10.y)))
names(sr)=unique(data_train_rest$char_10.y)
for(i in names(sr)){
  sr[i]=p[i]/(p[i]+n[i])
}
data_train_rest$char_10.ynew=rep(0, nrow(data_train_rest))
for(i in 1:nrow(data_train_rest)){
  data_train_rest$char_10.ynew[i]=sr[as.character(data_train_rest$char_10.y[i])]
}
data_train_rest$char_10.y=NULL

#Forward feature selection
sub_type1=regsubsets(data_train_type1$outcome~., data=data_train_type1, nbest=100, nvmax=30)
sub_rest=regsubsets(data_train_rest$outcome~., data=data_train_rest, nbest=100, nvmax=30)
summary(sub_type1)
summary(sub_rest)

#Sample the type 1 training data
X = c(1:dim(data_train_type1)[1])
ind = sample(X,50000)
data_tr_type1=data_train_type1[ind,]
data_test_type1=data_train_type1[-ind,]
data_test_type1=data_test_type1[1:5000,]
out_type1=data_test_type1$outcome
data_test_type1$outcome=NULL



#Sample rest of the types of training data
X = c(1:dim(data_train_rest)[1])
ind = sample(X,10000)
data_tr_rest=data_train_rest[ind,]
data_test_rest=data_train_rest[-ind,]
data_test_rest=data_test_rest[1:5000,]
#data_test_rest=data_test_rest[1,]
data_tr_rest$char_10.y=NULL
data_test_rest$char_10.y=NULL
out_rest=data_test_rest$outcome
data_test_rest$outcome=NULL


#Logistic regression on rest of activities
log_rest=glm(data_tr_rest$outcome ~ . ,family=binomial(link = "logit"), data=data_tr_rest)
predict_rest=predict(log_rest, data_test_rest, type="response")
for(i in 1:length(predict_rest)){
  if(predict_rest[i]>=0.5){
    predict_rest[i]=1
  }
  else
    predict_rest[i]=0
}
evaluate(out_rest, predict_rest)

#Logistic Regression on type 1
log_type1=glm(data_tr_type1$outcome ~ . , family=binomial(link="logit"),  data=data_tr_type1)
predict_type1=predict.glm(log_type1, data_test_type1, type="response")
for(i in 1:length(predict_type1)){
  if(predict_type1[i]>=0.4){
    predict_type1[i]=1
  }
  else
    predict_type1[i]=0
}
evaluate(out_type1, predict_type1)


#Decision tree on type 1
dtree_type1=rpart(data_tr_type1$outcome ~ . , method="class", data=data_tr_type1)
predict_dtree_type1=predict(dtree_type1, data_test_type1, type="vector")
for(i in 1:length(predict_dtree_type1)){
  if(predict_dtree_type1[i]==1){
    predict_dtree_type1[i]=0
  }
  else
    predict_dtree_type1[i]=1
}
acc_dtree_type1=(predict_dtree_type1==out_type1)
accuracy_dtree_type1=sum(acc_dtree_type1)/length(acc_dtree_type1)
accuracy_dtree_type1
evaluate(out_type1, predict_dtree_type1)

#Decision tree on rest
dtree_rest=rpart(data_tr_rest$outcome ~ . , method="class", data=data_tr_rest)
predict_dtree_rest=predict(dtree_rest, data_test_rest, type="vector")
for(i in 1:length(predict_dtree_rest)){
  if(predict_dtree_rest[i]==1){
    predict_dtree_rest[i]=0
  }
  else
    predict_dtree_rest[i]=1
}
evaluate(out_rest, predict_dtree_rest)

#Random Forest on type 1
rf_type1=randomForest(as.factor(data_tr_type1$outcome)~., data=data_tr_type1, ntree=1)
rf_type1.predict=predict(rf_type1, data_test_type1)
evaluate(out_type1, as.numeric(rf_type1.predict))

#Random Forest on rest
rf_rest=randomForest(as.factor(data_tr_rest$outcome)~., data=data_tr_rest, ntree=500, sampsize=3000)
rf_rest.predict=predict(rf_rest, data_test_rest)
evaluate(out_rest, as.numeric(rf_rest.predict))

#SVM for rest
#Polynomial
svm_rest_p <- svm(data_tr_rest$outcome ~ ., data = data_tr_rest, cost = 100, gamma = 1, kernel = 'polynomial')
svm_rest_p.predict <- predict(svm_rest_p, data_test_rest)
#Linear
svm_rest_l <- svm(data_tr_rest$outcome ~ ., data = data_tr_rest, cost = 100, kernel = 'linear')
svm_rest_l.predict <- predict(svm_rest_l, data_test_rest)
#Sigmoid
svm_rest_s <- svm(data_tr_rest$outcome ~ ., data = data_tr_rest, cost = 10, gamma = 100, coef0=0, kernel = 'sigmoid')
svm_rest_s.predict <- predict(svm_rest_s, data_test_rest)
#Radial
svm_rest_r <- svm(data_tr_rest$outcome ~ ., data = data_tr_rest, cost = 100, gamma = 1, kernel = 'radial')
svm_rest_r.predict <- predict(svm_rest_r, data_test_rest)

#SVM for type 1
#Polynomial
svm_type1_p <- svm(data_tr_type1$outcome ~ ., data = data_tr_type1, cost = 100, gamma = 1, kernel = 'polynomial')
svm_type1_p.predict <- predict(svm_type1_p, data_test_type1)
#Linear
svm_type1_l <- svm(data_tr_type1$outcome ~ ., data = data_tr_type1, cost = 100, kernel = 'linear')
svm_type1_l.predict <- predict(svm_type1_l, data_test_type1)
#Sigmoid
svm_type1_s <- svm(data_tr_type1$outcome ~ ., data = data_tr_type1, cost = 10, gamma = 100, coef0=0, kernel = 'sigmoid')
svm_type1_s.predict <- predict(svm_type1_s, data_test_type1)
#Radial
svm_type1_r <- svm(data_tr_type1$outcome ~ ., data = data_tr_type1, cost = 100, gamma = 1, kernel = 'radial')
svm_type1_r.predict <- predict(svm_type1_r, data_test_type1)


#Evaluation metrics
evaluate <- function(output, prediction){
  tp=sum(prediction & output)
  fp=sum(prediction & (!output))
  fn=sum((!prediction) & output)
  tn=sum((!prediction) & (!output))
  print(tp+fp+fn+tn)
  accuracy=(tp+tn)/(tp+fp+tn+fn)
  precision=(tp)/(tp+fp)
  recall=tp/(tp+fn)
  print(accuracy)
  print(precision)
  print(recall)
  f1=2*precision*recall/(precision + recall)
  print(f1)
}

######Random queries###########
#print(str(act_test))
#print(str(people))
unique_people=unique(act_train[c("people_id","outcome")])
unique_people_no=unique(act_train[c("people_id")])
print(dim(unique_people))
print(dim(unique_people_no))

#Class type of each attribute
lapply(act_train,class)
lapply(act_test,class)

#Calculating number of factors of each attribute
numFactors = apply(act_train, 2, function(x) nlevels(as.factor(x)))
print(numFactors)