#ML
# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

train <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/train.csv",header = TRUE)
store <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/store.csv",header = TRUE)
test <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/test.csv",header = TRUE)

head(train)
head(store)
head(test)
length(uniq.dates)

plot(train$Sales[which(train$Store==1)],pch=".",col=uniq.store[1],ylim=range(train$Sales,na.rm=T))
for(i in 2:length(uniq.store)-1){
  points(train$Sales[which(train$Store==i)],pch=".",col=uniq.store[i+1])
  }
  
# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
trains <- merge(train,store)

uniq.dates=unique(train$Date)
uniq.store=unique(train$Store)
ncol=length(uniq.store);nrow=length(uniq.dates);

#---------------- Sales vs Customers-----------

## Data in a data.frame
x1 <- train$Customers[which(train$Promo==1)]
x2 <- train$Sales[which(train$Promo==1)]
df <- data.frame(x1,x2)

## Use densCols() output to get density at each point
x <- densCols(x1,x2, colramp=colorRampPalette(c("black", "white")))
df$dens <- col2rgb(x)[1,] + 1L

## Map densities to colors
cols <-  colorRampPalette(c("#000099", "#00FEFF", "#45FE4F", 
                            "#FCFF00", "#FF9400", "#FF3100"))(256)
df$col <- cols[df$dens]

windows()
plot(x2~x1, data=df[order(df$dens),],pch=".",col=col, 
     xlab="Customers [#]", ylab="Sales [£]",main="Sales VS Customers Promo On")


## Data in a data.frame
x1 <- train$Customers[which(train$Promo==0)]
x2 <- train$Sales[which(train$Promo==0)]
df <- data.frame(x1,x2)

## Use densCols() output to get density at each point
x <- densCols(x1,x2, colramp=colorRampPalette(c("black", "white")))
df$dens <- col2rgb(x)[1,] + 1L

## Map densities to colors
cols <-  colorRampPalette(c("#000099", "#00FEFF", "#45FE4F", 
                            "#FCFF00", "#FF9400", "#FF3100"))(256)
df$col <- cols[df$dens]

windows()
plot(x2~x1, data=df[order(df$dens),],pch=".",col=col, 
     xlab="Customers [#]", ylab="Sales [£]",main="Sales VS Customers Promo Off")


## Data in a data.frame
x1 <- train$Customers
x2 <- train$Sales
df <- data.frame(x1,x2)

## Use densCols() output to get density at each point
x <- densCols(x1,x2, colramp=colorRampPalette(c("black", "white")))
df$dens <- col2rgb(x)[1,] + 1L

## Map densities to colors
cols <-  colorRampPalette(c("#000099", "#00FEFF", "#45FE4F", 
                            "#FCFF00", "#FF9400", "#FF3100"))(256)
df$col <- cols[df$dens]

windows()
plot(x2~x1, data=df[order(df$dens),],pch=".",col=col, 
     xlab="Customers [#]", ylab="Sales [£]",main="Sales VS Customers Global")

#---------------- Shops location vs Sales -----------

avg.sl.distON<-list()
avg.sl.distOFF<-list()
uniq.dist=unique(trains$CompetitionDistance)

for(i in 1:length(uniq.dist)){
  trains$Sales[trains$Sales==0]<-NA
  avg.sl.distON[[i]]=mean(trains$Sales[which(trains$CompetitionDistance==uniq.dist[i]&trains$Promo==1)],na.rm = T)
  avg.sl.distOFF[[i]]=mean(trains$Sales[which(trains$CompetitionDistance==uniq.dist[i]&trains$Promo==0)],na.rm = T)
}

avg.sl.diston=cbind(uniq.dist,do.call(rbind,avg.sl.distON))
avg.sl.distoff=cbind(uniq.dist,do.call(rbind,avg.sl.distOFF))
yrange=range(range(avg.sl.distoff[,2],na.rm=T),range(avg.sl.diston[,2],na.rm=T))
xrange=range(range(avg.sl.distoff[,1],na.rm=T),range(avg.sl.diston[,1],na.rm=T))

windows()
plot(x=avg.sl.distoff[,1],y=avg.sl.distoff[,2],col="red",ylim=yrange,xlim=xrange,pch=20,
     xlab="Distance [M]",ylab="Average Sales [£]")
abline(lm(avg.sl.distoff[,2]~avg.sl.distoff[,1]), col="red") # lowess line (x,y)
par(new=TRUE)
plot(x=avg.sl.diston[,1],y=avg.sl.diston[,2],col="blue",ylim=yrange,xlim=xrange,pch=20,
     xlab="",ylab="")
abline(lm(avg.sl.diston[,2]~avg.sl.diston[,1]), col="blue") # lowess line (x,y)
legend("topright",c("On Promo","Off Promo"),text.col=c("Blue","red"))
title("Average Sale by Competitor Distance")
grid()

#---------------- Avg Sales/Store by Promo-----------
avg.sl.proon<-list()
avg.sl.prooff<-list()
for( i in 1:length(uniq.store)){
train$Sales[train$Sales==0]<-NA
avg.sl.proon[[i]]=mean(train$Sales[which(train$Store==i&train$Promo==1)],na.rm = T)
avg.sl.prooff[[i]]=mean(train$Sales[which(train$Store==i&train$Promo==0)],na.rm = T)
}

avg.slON=do.call(rbind,avg.sl.proon)
avg.slOFF=do.call(rbind,avg.sl.prooff)
range=range(range(avg.slON),range(avg.slOFF))

windows()
plot(avg.slOFF,pch=20,col="red",ylim=range,xlab="Stores [#]",ylab="Average Sales [£]")
abline(h=mean(avg.slOFF), col="red") # lowess line (x,y)
par(new=TRUE)
plot(avg.slON,pch=20,col="Blue",ylim=range,xlab="",ylab="")
abline(h=mean(avg.slON), col="blue") # lowess line (x,y)

legend("topleft",c("On Promo","Off Promo"),text.col=c("Blue","red"))
title("Average Sale by Promotion Period")
grid()
            

#---------------- Avg Sales/Store by Assortment-----------
trains <- merge(train,store)
trains$Sales[trains$Sales==0]<-NA

avg.aa=mean(trains$Sales[which(trains$StoreType=="a"&trains$Assortment=="a"&train$Promo==0)],na.rm = T)
avg.ab=mean(trains$Sales[which(trains$StoreType=="a"&trains$Assortment=="b"&train$Promo==0)],na.rm = T)
avg.ac=mean(trains$Sales[which(trains$StoreType=="a"&trains$Assortment=="c"&train$Promo==0)],na.rm = T)
avg.ba=mean(trains$Sales[which(trains$StoreType=="b"&trains$Assortment=="a"&train$Promo==0)],na.rm = T)
avg.bb=mean(trains$Sales[which(trains$StoreType=="b"&trains$Assortment=="b"&train$Promo==0)],na.rm = T)
avg.bc=mean(trains$Sales[which(trains$StoreType=="b"&trains$Assortment=="c"&train$Promo==0)],na.rm = T)
avg.ca=mean(trains$Sales[which(trains$StoreType=="c"&trains$Assortment=="a"&train$Promo==0)],na.rm = T)
avg.cb=mean(trains$Sales[which(trains$StoreType=="c"&trains$Assortment=="b"&train$Promo==0)],na.rm = T)
avg.cc=mean(trains$Sales[which(trains$StoreType=="c"&trains$Assortment=="c"&train$Promo==0)],na.rm = T)
avg.da=mean(trains$Sales[which(trains$StoreType=="d"&trains$Assortment=="a"&train$Promo==0)],na.rm = T)
avg.db=mean(trains$Sales[which(trains$StoreType=="d"&trains$Assortment=="b"&train$Promo==0)],na.rm = T)
avg.dc=mean(trains$Sales[which(trains$StoreType=="d"&trains$Assortment=="c"&train$Promo==0)],na.rm = T)

a=cbind(avg.aa,avg.ab,avg.ac)
b=cbind(avg.ba,avg.bb,avg.bc)
c=cbind(avg.ca,avg.cb,avg.cc)
d=cbind(avg.da,avg.db,avg.dc)

store.avg<- t(rbind(a,b,c,d))
rownames(store.avg)<-c("A","B","C")
colnames(store.avg)<-c("A","B","C","D")
store.avg[is.na(store.avg)]<-0

windows()
barplot(store.avg, main="Average Sales by Store Type and Assortment",
        xlab="Store Type", ylab="Average Sales [£]",
        col=c("darkblue","red","orange","blue"),
        legend=rownames(store.avg))

#---------------- Unique time series ------
library(xts)
library(TTR)
library(lubridate)

#average by month for each year
xts.ts <- xts(trains$Sales,as.Date(trains$Date,origin=min(trains$Date)))
ts<-apply.monthly(xts.ts,sum,na.rm=T)
ts2<-c(as.numeric(t(ts)),rep(NA,5))
ts3<-matrix(ts2,nrow = 12,ncol = 3)
range=range(ts,na.rm=T)

windows()
plot(ts3[,1],type="l",col="darkgreen",xlab="Month",ylab="Total Sales [£]",ylim=range)
par(new=TRUE)
plot(ts3[,2],type="l",col="green",xlab="",ylab="",ylim=range)
par(new=TRUE)
plot(ts3[,3],type="l",col="lightgreen",xlab="",ylab="",ylim=range)
legend("topleft",c("2013","2014","2015"),text.col=c("darkgreen","green","lightgreen"))
title("Total Sales by Month")
grid()

#average by month for each year
xts.ts <- xts(trains$Customers,as.Date(trains$Date,origin=min(trains$Date)))
ts<-apply.monthly(xts.ts,sum,na.rm=T)
ts2<-c(as.numeric(t(ts)),rep(NA,5))
ts3<-matrix(ts2,nrow = 12,ncol = 3)
range=range(ts,na.rm=T)

windows()
plot(ts3[,1],type="l",col="darkblue",xlab="Month",ylab="Total Customers [#]",ylim=range)
par(new=TRUE)
plot(ts3[,2],type="l",col="blue",xlab="",ylab="",ylim=range)
par(new=TRUE)
plot(ts3[,3],type="l",col="lightblue",xlab="",ylab="",ylim=range)
legend("topleft",c("2013","2014","2015"),text.col=c("darkblue","blue","lightblue"))
title("Total Customers by Month")
grid()

#---------------- Avg Sales/Store by StateHoliday-----------
avg.sl.sopon<-list()
avg.sl.sopoff<-list()
avg.sl.sfpon<-list()
avg.sl.sfpoff<-list()

for( i in 1:length(uniq.store)){
  train$Sales[train$Sales==0]<-NA
  avg.sl.sopon[[i]]=mean(train$Sales[which(train$Store==i&train$SchoolHoliday==1&train$Promo==1)],na.rm = T)
  avg.sl.sopoff[[i]]=mean(train$Sales[which(train$Store==i&train$SchoolHoliday==1&train$Promo==0)],na.rm = T)
  avg.sl.sfpon[[i]]=mean(train$Sales[which(train$Store==i&train$SchoolHoliday==0&train$Promo==1)],na.rm = T)
  avg.sl.sfpoff[[i]]=mean(train$Sales[which(train$Store==i&train$SchoolHoliday==0&train$Promo==0)],na.rm = T)
}

avg.shpON=do.call(rbind,avg.sl.sopon)
avg.shpOFF=do.call(rbind,avg.sl.sopoff)
avg.shfpON=do.call(rbind,avg.sl.sfpon)
avg.shfpOFF=do.call(rbind,avg.sl.sfpoff)

range=range(range(avg.shpON),range(avg.shpOFF),range(avg.shfpON),range(avg.shfpOFF))

windows()
plot(avg.shpOFF,col="red",ylim=range,xlab="Stores [#]",ylab="Average Sales [£]",pch=20)
abline(h=mean(avg.shpOFF), col="red") # lowess line (x,y)
par(new=TRUE)
plot(avg.shpON,col="Blue",ylim=range,xlab="",ylab="",pch=20)
abline(h=mean(avg.shpON), col="blue") # lowess line (x,y)
par(new=TRUE)
plot(avg.shfpOFF,col="green",ylim=range,xlab="",ylab="",pch=20)
abline(h=mean(avg.shfpOFF), col="green") # lowess line (x,y)
par(new=TRUE)
plot(avg.shfpON,col="Darkblue",ylim=range,xlab="",ylab="",pch=20)
abline(h=mean(avg.shfpON), col="Darkblue") # lowess line (x,y)

legend("topleft",c("OFFSchoolHol-ONPromo","OFFSchoolHol-OFFPromo","ONSchoolHol-ONPromo","ONSchoolHol-OFFPromo"),
       text.col=c("Darkblue","green","Blue","red"))
title("Average Sale by School Holiday")
grid()

#---------------- Avg Sales/day of week by Promo-----------

x1 <- train$Sales[which(train$Promo==1&train$Sales!=0)]
x2 <- train$DayOfWeek[which(train$Promo==1&train$Sales!=0)]
df <- data.frame(x1,x2)
x <- densCols(x1,x2, colramp=colorRampPalette(c("black", "white")))
df$dens <- col2rgb(x)[1,] + 1L
cols <-  colorRampPalette(c("#00FEFF","#000099"))(256)
df$col <- cols[df$dens]

x3 <- train$Sales[which(train$Promo==0&train$Sales!=0)]
x4 <- train$DayOfWeek[which(train$Promo==0&train$Sales!=0)]
df2 <- data.frame(x3,x4)
xa <- densCols(x3,x4, colramp=colorRampPalette(c("black", "white")))
df2$dens <- col2rgb(xa)[1,] + 1L
cols2 <-  colorRampPalette(c("orange","red"))(256)
df2$col <- cols2[df2$dens]

range=range(range(x1),range(x3))

windows()
plot(x1~x2, data=df[order(df$dens),],pch=20,col=col,xlim=c(1,7), ylim=range,
     xlab="Days of Week", ylab="Sales [£]",main="Sales VS Days of Week")

par(new=TRUE)
plot(x3~x4, data=df2[order(df2$dens),],pch=20,col=col,xlim=c(1,7)+0.1,ylim=range, 
     xlab="", ylab="",xaxt="n")
legend("topleft",c("ONPromo","OFFPromo"),
       text.col=c("Blue","red"))

#---------------- Time series Data -------------
require(dplyr)
train <- read.csv("C:/Users/enricolo/Desktop/train.csv",header = TRUE)
test  <- read.csv("C:/Users/enricolo/Desktop/test.csv" ,header = TRUE)
dat = bind_rows(train,test)
rm(train,test)

dat$Date = as.Date(dat$Date)
dat$Open = dat$Open==1
dat$Promo = dat$Promo==1
dat$SchoolHoliday = dat$SchoolHoliday==1

head(dat)

# FIT Model
train = dat[dat$split=='train',]
train = train[train$Sales>0,]
preds=c('Promo')
mdl = train %>% group_by_(.dots=preds) %>% summarise(predSales=median(Sales)) %>% ungroup()
predict.dplyr = function(mdl,newdata) {
  x=(newdata %>% left_join(mdl,by=preds) %>% select(Id,predSales) %>% rename(Sales=predSales))$Sales
  x[is.na(x)]=0
  x
}
## Predict
dat$predSales=predict.dplyr(mdl,newdata=dat) 

# Graph
s=1081
gd=dat$store==s 
x=dat[gd,]
## Find promo periods and holidays
gd=x$stateHoliday!='0'
sth = data.frame(date=x$date[gd],stateHoliday=x$stateHoliday[gd])
sth$color = plyr::mapvalues(sth$stateHoliday,from=c('a','b','c'),to=c('black','blue','red'))
promos = data.frame(
start = x$date[x$promo-lag(x$promo,default=0)>0], 
end = c(x$date[x$promo-lag(x$promo,default=0)<0],x$date[nrow(x)]))
## Make the graph
require(xts)
require(dygraphs)
dyEvents = function(x,date,label=NULL,labelLoc='bottom',color='black',strokePattern='dashed') {
  for (i in 1:length(date)) x = x %>% dyEvent(date[i],label[i],labelLoc,color[i],strokePattern)
  x
}
dyShadings = function(x,from,to,color="#EFEFEF") {
  for (i in 1:length(from)) x = x %>% dyShading(from[i],to[i],color)
  x
}
y=cbind(sales=xts(x$sales,x$date),
        predSales=xts(x$predSales,x$date))
dygraph(y, main = "Real & Pred Sales", group = "q", width=800) %>% 
  dySeries('sales', drawPoints = TRUE, color = 'blue') %>%
  dySeries('predSales', drawPoints = TRUE, color = 'green') %>%
  dyRoller(rollPeriod=1) %>%
  dyShadings(promos$start,promos$end) %>%
  dyEvents(sth$date,color=sth$color) %>% 
  dyRangeSelector(dateWindow=as.Date(c('2014-01-01','2014-06-01')))

#---------------- KNN ----------
library(e1071)
library(knn)
library(rpart)
library(randrandomForest)

train <- read.csv("C:/Users/enricolo/Desktop/train.csv",header = TRUE)
# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
str(train)
summary(train)
# looking at only stores that were open in the train set
train <- train[ which(train$Open=='1'&train$Sales!='0'),]
# seperating out the elements of the date column for the train set
train$month <- as.integer(format(as.Date(train$Date), "%m"))
train$year <- as.integer(format(as.Date(train$Date), "%y"))
train$day <- as.integer(format(as.Date(train$Date), "%d"))
# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8)]

test <- read.csv("C:/Users/enricolo/Desktop/test.csv",header = TRUE)
# seperating out the elements of the date column for the test set
test$month <- as.integer(format(as.Date(test$Date), "%m"))
test$year <- as.integer(format(as.Date(test$Date), "%y"))
test$day <- as.integer(format(as.Date(test$Date), "%d"))
# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,5:19)]
feature.names

for (f in feature.names) {
 if (class(train[[f]])=="character") {
   levels <- unique(c(train[[f]], test[[f]]))
   train[[f]] <- as.integer(factor(train[[f]], levels=levels))
   test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
 }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]

RMPSE<- function(preds, dtrain) {
 labels <- getinfo(dtrain, "label")
 elab<-exp(as.numeric(labels))-1
 epreds<-exp(as.numeric(preds))-1
 err <- sqrt(mean((epreds/elab-1)^2))
 return(list(metric = "RMPSE", value = err))
}
nrow(train)
h<-sample(nrow(train),10000)

library(xgboost)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)

param <- list(  objective           = "reg:linear", 
               booster = "gbtree",
               eta                 = 0.02, # 0.06, #0.01,
               max_depth           = 10, #changed from default of 8
               subsample           = 0.8, # 0.7
               colsample_bytree    = 0.7 # 0.7
               #num_parallel_tree   = 2
               # alpha = 0.0001, 
               # lambda = 1
)

clf <- xgb.train(params              = param, 
                 data                = dtrain, 
                 nrounds             = 10, #300, #280, #125, #250, # changed from 300
                 verbose             = 0,
                 early.stop.round    = 10,
                 watchlist           = watchlist,
                 maximize            = FALSE,
                 feval=RMPSE
)
pred1 <- exp(predict(clf, data.matrix(test[,1:7]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)

#---------------- Random forest-----
# Load Dependeicies
#install.packages("caret", dependencies = c("Depends", "Suggests"))
#devtools::install_github('topepo/caret/pkg/caret')
library(caret)
library(randomForest)
library(readr)
library(lubridate)
library(plyr)

# Set seed
set.seed(1337)

train <- read.csv("C:/Users/enricolo/Desktop/train.csv",header = TRUE)
store <- read.csv("C:/Users/enricolo/Desktop/store.csv",header = TRUE)
test <- read.csv("C:/Users/enricolo/Desktop/test.csv",header = TRUE)

# Merge Store stuff
train <- merge(train, store, by=c('Store'))
test <- merge(test, store, by=c('Store'))
test <- arrange(test,Id)

# Only evaludated on Open days
train <- train[ which(train$Open=='1'),]
train$Open <- NULL
test$Open <- NULL

# Set missing data to 0
train[is.na(train)] <- 0
test[is.na(test)] <- 0

# Date stuff
train$Date <- ymd(train$Date)
test$Date <- ymd(test$Date)
train$month <- as.factor(month(train$Date))
test$month <- as.factor(month(test$Date))
train$year <- as.factor(year(train$Date))
test$year <- as.factor(year(test$Date))

#Factorize stuff
train$DayOfWeek <- as.factor(train$DayOfWeek)
test$DayOfWeek <- as.factor(test$DayOfWeek)
train$Open <- as.factor(train$Open)
test$Open <- as.factor(test$Open)
train$Promo <- as.factor(train$Promo)
test$Promo <- as.factor(test$Promo)
train$SchoolHoliday <- as.factor(train$SchoolHoliday)
test$SchoolHoliday <- as.factor(test$SchoolHoliday)

# always 1 for training and test
train$StateHoliday <- NULL
test$StateHoliday <- NULL

# Factorize store stuff
train$StoreType <- as.factor(train$StoreType)
test$StoreType <- as.factor(test$StoreType)
train$Assortment <- as.factor(train$Assortment)
test$Assortment <- as.factor(test$Assortment)
train$CompetitionDistance <- as.numeric(train$CompetitionDistance)
test$CompetitionDistance <- as.numeric(test$CompetitionDistance)

# target variables
train$Sales <- as.numeric(train$Sales)
train$Customers <- NULL #as.numeric(train$Customers)

fitControl <- trainControl(method="cv", number=3, verboseIter=T)
rfFit <- train(Sales ~., 
               method="rf", data=train, ntree=50, importance=TRUE,
               sampsize=100000,
               do.trace=10, trControl=fitControl)

pred <- predict(rfFit, test)
submit = data.frame(Id = test$Id, Sales = pred)

#---------------- Results/Parameters Plot ------------
ML1 <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/ML1.csv",header = TRUE)
ML2 <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/ML2.csv",header = TRUE)
ML3 <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/ML3.csv",header = TRUE)

range=range(range(ML1[,2]),range(ML2[,2]),range(ML3[,2]))
windows()
plot(ML1[,2],pch=20, col="darkblue",xlab = "Store ID",ylab="Predicted Sales",ylim=range)
points(ML2[,2],pch=20, col="orange",xlab="",ylab="",ylim=range)
points(ML3[,2],pch=20, col="RED",xlab="",ylab="",ylim=range)
title("Predicted Sales VS Algorithms")
legend("topleft",c("Bayesian Ridge Regression","Random Forest Regression","Knn"),
       text.col=c("darkblue","orange","red"))
grid()

range=range(range(ML2[,2]),range(ML3[,2]))
windows()
plot(ML2[,2],pch=20, col="darkblue",xlab = "Store ID",ylab="Predicted Sales",ylim=range)
points(ML3[,2],pch=20, col="RED",xlab="",ylab="",ylim=range)
title("Predicted Sales VS Algorithms")
legend("topleft",c("Random Forest Regression","KNN Regression"),
       text.col=c("darkblue","red"))
grid()


RF <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/randomForest_testing_results.csv",header = TRUE)
KN <- read.csv("C:/Users/enricolo/Desktop/STUDIO/7 - Msc Data Science - City/3 - Machine Learning/Coursework/Project/k_nearest_testing_results.csv",header = TRUE)

length(RF$Error)
range=range(range(RF$Error),range(KN$Error))

windows()
plot(RF$Error,pch=20, col="darkblue",xlab = "Trials",ylab="Mean Squared Error",ylim=range)
points(KN$Error,pch=20, col="red",xlab="",ylab="",ylim=range)
title("MSE VS Algorithms")
legend("topleft",c("Random Forest Regression","KNN Regression"),
       text.col=c("darkblue","red"))
grid()

head(RF)
windows()
par(mfrow=c(1,4))
plot(RF$max_depth,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "Max Depth")
grid()
plot(RF$n_estimators,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "Num Estimators")
grid()
plot(RF$criterion,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "Criterion")
grid()
plot(RF$max_features,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "Max Features")
grid()

head(BR)
windows()
par(mfrow=c(1,4))
plot(BR$n_iter,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "# Iterations")
grid()
plot(BR$alpha_2,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "Alpha 2")
grid()
plot(BR$tol,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "Tol")
grid()
plot(BR$alpha_1,pch=20, col="darkblue",xlab = "Trials",ylab="#",main = "Alpha 1")
grid()

vwReg <- function(formula, data, title="", B=1000, shade=TRUE, shade.alpha=.1, spag=FALSE, spag.color="darkblue", mweight=TRUE, show.lm=TRUE, show.median = TRUE, median.col = "white", shape = 21, show.CI=TRUE, method=loess, bw=FALSE, slices=200, palette=colorRampPalette(c("#FFEDA0", "#DD0000"), bias=2)(20), ylim=NULL, quantize = "continuous",  add=FALSE, ...) {
  IV <- all.vars(formula)[2]
  DV <- all.vars(formula)[1]
  data <- na.omit(data[order(data[, IV]), c(IV, DV)])
  
  if (bw == TRUE) {
    palette <- colorRampPalette(c("#EEEEEE", "#999999", "#333333"), bias=2)(20)
  }
  
  print("Computing boostrapped smoothers ...")
  newx <- data.frame(seq(min(data[, IV]), max(data[, IV]), length=slices))
  colnames(newx) <- IV
  l0.boot <- matrix(NA, nrow=nrow(newx), ncol=B)
  
  l0 <- method(formula, data)
  for (i in 1:B) {
    data2 <- data[sample(nrow(data), replace=TRUE), ]
    data2 <- data2[order(data2[, IV]), ]
    if (class(l0)=="loess") {
      m1 <- method(formula, data2, control = loess.control(surface = "i", statistics="a", trace.hat="a"), ...)
    } else {
      m1 <- method(formula, data2, ...)
    }
    l0.boot[, i] <- predict(m1, newdata=newx)
  }
  
  # compute median and CI limits of bootstrap
  library(plyr)
  library(reshape2)
  CI.boot <- adply(l0.boot, 1, function(x) quantile(x, prob=c(.025, .5, .975, pnorm(c(-3, -2, -1, 0, 1, 2, 3))), na.rm=TRUE))[, -1]
  colnames(CI.boot)[1:10] <- c("LL", "M", "UL", paste0("SD", 1:7))
  CI.boot$x <- newx[, 1]
  CI.boot$width <- CI.boot$UL - CI.boot$LL
  
  # scale the CI width to the range 0 to 1 and flip it (bigger numbers = narrower CI)
  CI.boot$w2 <- (CI.boot$width - min(CI.boot$width))
  CI.boot$w3 <- 1-(CI.boot$w2/max(CI.boot$w2))
  
  
  # convert bootstrapped spaghettis to long format
  b2 <- melt(l0.boot)
  b2$x <- newx[,1]
  colnames(b2) <- c("index", "B", "value", "x")
  
  library(ggplot2)
  library(RColorBrewer)
  
  # Construct ggplot
  # All plot elements are constructed as a list, so they can be added to an existing ggplot
  
  # if add == FALSE: provide the basic ggplot object
  p0 <- ggplot(data, aes_string(x=IV, y=DV)) + theme_bw()
  
  # initialize elements with NULL (if they are defined, they are overwritten with something meaningful)
  gg.tiles <- gg.poly <- gg.spag <- gg.median <- gg.CI1 <- gg.CI2 <- gg.lm <- gg.points <- gg.title <- NULL
  
  if (shade == TRUE) {
    quantize <- match.arg(quantize, c("continuous", "SD"))
    if (quantize == "continuous") {
      print("Computing density estimates for each vertical cut ...")
      flush.console()
      
      if (is.null(ylim)) {
        min_value <- min(min(l0.boot, na.rm=TRUE), min(data[, DV], na.rm=TRUE))
        max_value <- max(max(l0.boot, na.rm=TRUE), max(data[, DV], na.rm=TRUE))
        ylim <- c(min_value, max_value)
      }
      
      # vertical cross-sectional density estimate
      d2 <- ddply(b2[, c("x", "value")], .(x), function(df) {
        res <- data.frame(density(df$value, na.rm=TRUE, n=slices, from=ylim[1], to=ylim[2])[c("x", "y")])
        #res <- data.frame(density(df$value, na.rm=TRUE, n=slices)[c("x", "y")])
        colnames(res) <- c("y", "dens")
        return(res)
      }, .progress="text")
      
      maxdens <- max(d2$dens)
      mindens <- min(d2$dens)
      d2$dens.scaled <- (d2$dens - mindens)/maxdens   
      
      ## Tile approach
      d2$alpha.factor <- d2$dens.scaled^shade.alpha
      gg.tiles <-  list(geom_tile(data=d2, aes(x=x, y=y, fill=dens.scaled, alpha=alpha.factor)), scale_fill_gradientn("dens.scaled", colours=palette), scale_alpha_continuous(range=c(0.001, 1)))
    }
    if (quantize == "SD") {
      ## Polygon approach
      
      SDs <- melt(CI.boot[, c("x", paste0("SD", 1:7))], id.vars="x")
      count <- 0
      d3 <- data.frame()
      col <- c(1,2,3,3,2,1)
      for (i in 1:6) {
        seg1 <- SDs[SDs$variable == paste0("SD", i), ]
        seg2 <- SDs[SDs$variable == paste0("SD", i+1), ]
        seg <- rbind(seg1, seg2[nrow(seg2):1, ])
        seg$group <- count
        seg$col <- col[i]
        count <- count + 1
        d3 <- rbind(d3, seg)
      }
      
      gg.poly <-  list(geom_polygon(data=d3, aes(x=x, y=value, color=NULL, fill=col, group=group)), scale_fill_gradientn("dens.scaled", colours=palette, values=seq(-1, 3, 1)))
    }
  }
  
  print("Build ggplot figure ...")
  flush.console()
  
  
  if (spag==TRUE) {
    gg.spag <-  geom_path(data=b2, aes(x=x, y=value, group=B), size=0.7, alpha=10/B, color=spag.color)
  }
  
  if (show.median == TRUE) {
    if (mweight == TRUE) {
      gg.median <-  geom_path(data=CI.boot, aes(x=x, y=M, alpha=w3^3), size=.6, linejoin="mitre", color=median.col)
    } else {
      gg.median <-  geom_path(data=CI.boot, aes(x=x, y=M), size = 0.6, linejoin="mitre", color=median.col)
    }
  }
  
  # Confidence limits
  if (show.CI == TRUE) {
    gg.CI1 <- geom_path(data=CI.boot, aes(x=x, y=UL), size=1, color="red")
    gg.CI2 <- geom_path(data=CI.boot, aes(x=x, y=LL), size=1, color="red")
  }
  
  # plain linear regression line
  if (show.lm==TRUE) {gg.lm <- geom_smooth(method="lm", color="darkgreen", se=FALSE)}
  
  gg.points <- geom_point(data=data, aes_string(x=IV, y=DV), size=1, shape=shape, fill="white", color="black")        
  
  if (title != "") {
    gg.title <- theme(title=title)
  }
  
  
  gg.elements <- list(gg.tiles, gg.poly, gg.spag, gg.median, gg.CI1, gg.CI2, gg.lm, gg.points, gg.title, theme(legend.position="none"))
  
  if (add == FALSE) {
    return(p0 + gg.elements)
  } else {
    return(gg.elements)
  }
}

x1 <- seq(1,length(RF$Error),1)
y1 <- RF$Error
df1 <- data.frame(x1, y1)

p3 <- vwReg(y1~x1,df1,shade.alpha=0.005,slices=400,ylim = range,
palette=colorRampPalette(c("white","green","yellow","red"),bias=1)(20));p3

x <- seq(1,length(KN$Error),1)
y <- KN$Error
df <- data.frame(x, y)

p1 <- vwReg(y~x,df, spag=TRUE, shade=FALSE,ylim = range);p1
