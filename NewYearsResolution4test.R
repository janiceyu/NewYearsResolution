tweets_Unclean <- read.csv(file="C:\Users\Janice\OneDrive\Documents\Data\CKME136\NewYearsResolution", header=TRUE)

str(tweets_Unclean)

#remove retweets
tweets <- tweets_Unclean[tweets_Unclean$retweet_count==0,]

#checking filtered % = 16% of tweets filtered out that was based on Retweets
1-nrow(tweets)/nrow(tweets_Unclean)

#install text mining packages
install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)
install.packages("tidytext")
library(tidytext)
install.packages("dplR")
library(dplR)
library(dplyr)
#start a corpus from text from tweets
myCorpus = Corpus(VectorSource(tweets$text))
#make a backup copy
myCorpusCopy <- myCorpus

myCorpus <- myCorpusCopy
#build a function to process tweets****************************************


# convert to lower case
CorpusTemp = tm_map(myCorpus, content_transformer(tolower))
inspect(CorpusTemp)

# remove stop words
CorpusTemp  <- tm_map(CorpusTemp, removeWords, c(stopwords("english"), 'newyearsresolution'))
inspect(CorpusTemp)

# remove punctuation
CorpusTemp = tm_map(CorpusTemp, removePunctuation)
inspect(CorpusTemp)

# remove numbers
CorpusTemp = tm_map(CorpusTemp, removeNumbers)
inspect(CorpusTemp)


# remove URL Function 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
# transform corpus by removingURL
CorpusTemp  <- tm_map(CorpusTemp, content_transformer(removeURL))
inspect(CorpusTemp)

# Remove duplicate letters of >2 repeat
removeDuplicateChar <- function(x) gsub("([[:alpha:]])\\1{2,}", "\\1", x)
CorpusTemp  <- tm_map(CorpusTemp, content_transformer(removeDuplicateChar))

inspect(CorpusTemp)

#Get Sentiment Analysis through Bing Lexicon
Bing_Lex <- get_sentiments("bing")

#
myCorpusTokenized <- lapply(CorpusTemp, scan_tokenizer)
 

#check 5 tweet results from processed corpus
for (i in c(1:4, 160)) {
  #label lines [1],[2] etc...
  cat(paste0("[", i, "] "))
  #paste corpus contents
  writeLines(strwrap(as.character(CorpusTemp[[i]]), 60))
}

#Compared to unprocessed corpus LOOKS THE SAME WHAT???
for (i in c(1:4, 160)) {
  #label lines [1],[2] etc...
  cat(paste0("[", i, "] "))
  #paste corpus contents
  writeLines(strwrap(as.character(myCorpusCopy[[i]]), 60))
}

# stem document LAST STEP FOR MODELLING
CorpusTemp  <- tm_map(CorpusTemp, stemDocument)
inspect(CorpusTemp)
