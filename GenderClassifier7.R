#initializing
install.packages("quanteda")
install.packages("caret")
install.packages("e1071")
install.packages("doSNOW")
install.packages("irlba")
install.packages("dplr")
libs <- c("tm","plyr", "class", "dplyr", "stringr", "e1071", "ggplot2", "irlba", "dplR")
library(caret)
library(quanteda)
library("doSNOW")
library(tm)
lapply(libs, require, character.only = TRUE)

#set Options - tells R to keep character variables 
#as they are rather than convert to factors
options(StringsAsFactors = FALSE)

#load files
tweets_Unclean <- read.csv(file="C:/Users/Janice/OneDrive/Documents/Data/CKME136/NewYearsResolution/NewYearsPresent/New-years-resolutions-DFE.csv", header=TRUE)

Female <- tweets_Unclean[tweets_Unclean$gender == 'female',"text"] #tweets for females

Male <- tweets_Unclean[tweets_Unclean$gender == 'male',"text"] #tweets for males

#check number of rows length(Female)+length(Male)

#check proportion of males and females
prop.table(table(tweets_Unclean$gender))

#clean text function
Clean <- function(clean_tweet){
  #lower case 
  clean_tweet = tolower(clean_tweet)
  
  # Regex pattern for removing stop words
  stop_pattern <- paste0("\\b(", paste0(stopwords("en"), collapse="|"), ")\\b")
  clean_tweet <- gsub(stop_pattern, "", clean_tweet)
  
  # Remove all #hashtags and @mentions
  clean_tweet <- gsub("(?:#|@)[a-zA-Z0-9_]+ ?", "", clean_tweet)
  
  #remove all ampersand 
  clean_tweet <- gsub("&amp;","", clean_tweet)
  
  #remove URL
  clean_tweet = gsub("http[^[:space:]]*", "", clean_tweet)
  
  #remove punctuation
  clean_tweet = gsub("[[:punct:]]", "", clean_tweet)
  
  #remove retweet
  clean_tweet = gsub("rt ", "", clean_tweet) 
  
  #remove >2 repeating letters
  clean_tweet = gsub("([[:alpha:]])\\1{2,}", "\\1", clean_tweet)
  
  #remove numbers
  clean_tweet = gsub("[[:digit:]]", "", clean_tweet)
  
  #remove all line breaks
  clean_tweet = str_replace_all(clean_tweet, "[\r\n]" , " ")
  
  #remove tabs
  clean_tweet = gsub("[ \t]{2,}", "", clean_tweet)
  
  #remove >2 repeating letters
  clean_tweet = gsub("([[:alpha:]])\\1{2,}", "\\1", clean_tweet)
  
  #remove newyearsresolution because that's the repeated hashtag
  clean_tweet = gsub("(newyearsresolution)|(new year)|(resolution)", "" , clean_tweet)
  
  #replace all non-alpha numeric characters just in case
  clean_tweet = str_replace_all(clean_tweet, "[^[:alnum:]]", " ")
  
  #remove extra white space
  clean_tweet = gsub("^\\s+|\\s+$", "", clean_tweet) 
  clean_tweet <- gsub(" {2,}", " ", clean_tweet)
  
  return(clean_tweet)
}

#build Term Document Matrix
generateTDM <- function(gender) {
  cor.cl <- Clean(gender) #run cleaning function on tweet 
  myCorpus = Corpus(VectorSource(cor.cl)) #put the tweet into a corpus 
  myCorpus  <- tm_map(myCorpus, stemDocument) #if stemming is required. I found that stemming decrease model accuracy by 8%
  t.tdm <- TermDocumentMatrix(myCorpus)
  #t.tdm <- removeSparseTerms(t.tdm,0.98) remove infrequent terms if required since these are tweets generally limited character length so I'm not going to use this
  result <- list(name=deparse(substitute(gender)), tdm = t.tdm) #lists tdm by female and male
}

TDM_Compile <-list(generateTDM(Female), generateTDM(Male))
str(TDM_Compile)


#attach Gender
bindgenderToTDM <- function (tdm1){
  g.mat <-t(data.matrix(tdm1[["tdm"]])) #transpose the element
  g.df <- as.data.frame(g.mat, stringAsFactors = FALSE) #convert into dataframe 
  g.df <- cbind(g.df, rep(tdm1[["name"]], nrow(g.df))) #attach gender "name" to each row
  colnames(g.df)[ncol(g.df)] <- "targetgender" #labeling last col of dataframe as target gender
  return(g.df)
}

genderTDM <-lapply(TDM_Compile,bindgenderToTDM)

str(genderTDM)

#stack 
#row binding and since these dataframes may have different terms, fill NA when 
#there's a term in one but doesn't appear in the other
tdm.stack <- do.call(rbind.fill, genderTDM) 
#Look for NA and replace with 0
tdm.stack[is.na(tdm.stack)] <--0
#checking number rows
nrow(tdm.stack)
#checking number of unique terms
ncol(tdm.stack)
View(tdm.stack)



#hold-out
#take out sample to train 70%
# Use caret package to create a 70%/30% stratified split. 
# Set seed for reproducibility.
set.seed(1)
train.idx <- createDataPartition(tdm.stack$targetgender, times = 1, p = 0.7, list = FALSE)
#create training and test index
train <-tdm.stack[train.idx,]
test <- tdm.stack[-train.idx,]


# Verify proportions.
prop.table(table(train$targetgender))
prop.table(table(test$targetgender))

# Use caret package to create stratified folds for 10-fold cross validation repeated 
# 3 times (i.e., create 30 random stratified samples)
cv.folds <- createMultiFolds(train$targetgender, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)


#Model #1 Using KNN and using a confusion matrix to test overall accuracy

tdm.gender <-tdm.stack[,"targetgender"] #list all rows but only the column label
tdm.stack.n1 <- tdm.stack[,!colnames(tdm.stack) %in% "targetgender"] #identify all the rows and columns except the ones in label column

knn.pred <- knn(tdm.stack.n1[train.idx, ], tdm.stack.n1[-train.idx,], tdm.gender[train.idx]) 

#check accuracy
conf.mat <- table("Predictions" = knn.pred, Actual = tdm.gender[test.idx])
conf.mat
(accurancy <-sum(diag(conf.mat)/length(test.idx) *100))
#57% accuracy overall

#Next following steps would apply a term frequency-inverse document frequency. It is a numerical statistic
#that is intended to reflect how important a word is to a document in a collection of a corpus. Aim is to improve the model

# Function for calculating relative term frequency (TF)
term.frequency <- function (row){
  row / sum(row)
}

# Function for calculating inverse document frequency (IDF) [calculate by columns looking at terms, for each column how many tweets are there]
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  log10(corpus.size/doc.count)
}

#function for calculating TF-IDF
tf.idf <- function (tf, idf) {
  tf * idf
}


#build matrix
train.tokens.matrix <- as.matrix(tdm.stack.n1[train.idx, ])


# first step, normalize all documents with TF**test ** 
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency) #apply term frequency to columns 

# second step, calculate the IDF vector that we will use both 
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)

# Lastly, calculate TF-IDF for training corpus.
train.tokens.tfidf <-  apply(train.tokens.df , 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf)

# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf)


# Check for incomplete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf)) 

train <- tdm.stack[train.idx, ]
train$text[incomplete.cases]
train$text

# Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

train.tokens.tfidf

# Make a clean data frame using the same process as before.
train.tokens.tfidf.df <- cbind(gender = train$targetgender, data.frame(train.tokens.tfidf))

names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

#TF-IDF improves accuracy by neutralizing document lengths and words that appear on documents would be weighted based on frequency
train.tokens.tfidf.df

#Model #1 Again Using **KNN** and TDIF using a confusion matrix to test overall accuracy

knn.pred2 <- knn(train.tokens.tfidf, tdm.stack.n1[test.idx,], tdm.gender[train.idx]) 

#check accuracy
conf.mat <- table("Predictions" = knn.pred2, Actual = tdm.gender[test.idx])
conf.mat
(accurancy <-sum(diag(conf.mat)/length(test.idx) *100))
#63% improvement with using TF-IDF rather than counts from a dataframe

#CROSS VALIDATION - **KNN**
#create a cluster to work on 2 logical cores
cl<- makeCluster(3,type="SOCK")
registerDoSNOW(cl)

# As our data is  use a single decision 
# tree algorithm 
#a feature extraction to shrime the size of our data

knn.cv <- train(gender~., data = train.tokens.tfidf.df, method = "knn", trControl = cv.cntrl, tuneLength = 7)

# Total time of execution was
total.time <- Sys.time() - start.time
total.time

knn.cv

#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was cp = 0.01947308. Accuracy of 58.6%


#Creating 2nd Model - Tree algorithm
#Time the code execution
start.time <- Sys.time()

#create a cluster to work on 2 logical cores
cl<- makeCluster(2,type="SOCK")
registerDoSNOW(cl)

train.tokens.tfidf.df
# As our data is  use a single decision 
# tree algorithm 
#a feature extraction to shrime the size of our data
rpart.cv <- train(gender~., data = train.tokens.tfidf.df, method = "rpart", trControl = cv.cntrl, tuneLength = 7)

# Total time of execution was
total.time <- Sys.time() - start.time
total.time

rpart.cv
#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was cp = 0.01947308. Accuracy of 58.6%

#Tree Model in practice 
rpart1 <- rpart(gender ~ ., data = train.tokens.tfidf.df)
rpart.predict <- predict(rpart1, tdm.stack.n1[test.idx,])
conf.mat2 <- table("Predictions" = rpart.predict, Actual = tdm.gender[test.idx])
(accurancy <-sum(diag(conf.mat2)/length(test.idx) *100))
#49.57% overall accuracy 


#Create 3rd Model Using Naive Bayes~~~~~
#create a cluster to work on 2 logical cores
cl<- makeCluster(2,type="SOCK")
registerDoSNOW(cl)

train.tokens.tfidf.df
# As our data is non-trivial in size at this point, use a single decision 
# naive bayes model for training 
nb.cv <- train(gender~., data = train.tokens.tfidf.df, method = "nb", trControl = cv.cntrl, tuneLength = 7)

# Total time of execution was
total.time <- Sys.time() - start.time
total.time

#caret package to predict
N.pred1 <- predict(nb.cv, newdata = tdm.stack.n1[test.idx,])

nb.cv
#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were fL = 0, usekernel = TRUE and adjust = 1. Accuracy of 48.6%


#Using e1071 package to test Naive bayes
NB.model <- naiveBayes(gender ~ ., data = train.tokens.tfidf.df)
NB.pred <- predict(NB.model, tdm.stack.n1[test.idx,])
conf.mat2 <- table("Predictions" = NB.pred, Actual = tdm.gender[test.idx])
(accurancy <-sum(diag(conf.mat2)/length(test.idx) *100))
#49.2% overall accuracy

#EXPLORATORY ANALYSIS OF DATA

#How to count frequent words in female tweets

#mystopwords adding more to filter for better analysis removing tiny text
lexicon

tdmgenerate <- function(tweet){ #building function to create quick term document matrix
  cor.cl <- Clean(tweet) #cleaning tweets
  Corpus = Corpus(VectorSource(cor.cl)) #creating corpus with just specific gender tweets
  tdm <- TermDocumentMatrix(Corpus)
  return(tdm)
}

#creates a data frame of frequent words 
freq.table <- function(x){
  term.freq <- rowSums(as.matrix(tdmgenerate(x)))
  term.freq <- subset(term.freq, term.freq >=10)
  words.df <- data.frame(word = names(term.freq), freq = term.freq)
  words.df$word <- as.character(words.df$word) #change words from factors char
  return(words.df)
}
library(tidytext)
library(dplR)

my_stop_words <- c(stop_words,"stop")
my_stop_words

#FEMALE TWEETS ANALYSIS
  f.table <- freq.table(tdmgenerate(Female)) #frequency table of words for female tweets
  Filtered.F <- freq.table(Female) %>% anti_join(my_stop_words, by ="word") #filter stop words from tidytext

#compare to before filtering with tidytext went from 43 words --> 13 words 
nrow(Filtered.F)
nrow(f.table)

# Relevel the words and order by higher to lower frequency when graphed 
Filtered.F$word <-factor(Filtered.F$word, levels=Filtered.F[order(Filtered.F$freq),"word"])

#plot frequent Female terms in regards to new year resolutions                        
ggplot(Filtered.F, aes(x = word, y = freq, fill ="pink")) + geom_bar(stat = "identity") +
  xlab("Terms") + ylab("Count") + coord_flip() + ggtitle("Top Female Words")

#MALE TWEETS ANALYSIS
m.table <- freq.table(Male) #frequency table of words for female tweets
Filtered.m <- freq.table(Male) %>% anti_join(stop_words, byr ="word") #filter stop words from tidytext

## Relevel the words by frequency of words 
Filtered.m$word <-factor(Filtered.m$word, levels=Filtered.m[order(Filtered.m$freq),"word"])

#plot frequent male terms in regards to new year resolutions                        
ggplot(Filtered.m, aes(x = word, y = freq)) + geom_bar(stat="identity") +
  xlab("Terms") + ylab("Count") + coord_flip() + ggtitle("Top Male Words")

#build TDM for Female tweets
generateTDM2 <- function(gender) {
  cor.cl <- Clean(gender) #run cleaning function on tweet 
  myCorpus = Corpus(VectorSource(cor.cl)) #put the tweet into a corpus 
  myCorpus  <- tm_map(myCorpus, stemDocument) #if stemming is required. I found that stemming decrease model accuracy by 8%
  t.tdm <- TermDocumentMatrix(myCorpus)
return (t.tdm)
}


#Find association for stop for Females
findAssocs(generateTDM2(Female), "stop", 0.2)

#Find association for stop for Males
findAssocs(generateTDM2(Male), "stop", 0.2)

#Find association for supper
findAssocs(generateTDM2(tweets_Unclean$text), "supper", 0.2)
#Seems like Highest association with Supper is Supper Club tweets 


# calculate the frequency of female words and sort it by frequency
word.freq <- sort(Filtered.M, decreasing = T)
# colors
pal <- brewer.pal(9, "Blues")
pal <- pal[-(1:4)]

library("wordcloud")

wordcloud(words = names(word.freq), freq = word.freq, min.freq = 3,
          random.order = F, colors = pal)


