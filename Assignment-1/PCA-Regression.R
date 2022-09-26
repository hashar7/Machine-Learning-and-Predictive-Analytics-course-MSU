data <- read.csv("test_sample.csv", sep = ' ')
X<-data.matrix(data[,2:492])
Y<-data[,1]
rSquared<-sapply(1:491,function(z) summary(lm(Y~.,data=data.frame(Y=Y,X[,1:z])))$r.squared)
N.orig<-min(which(rSquared > 0.9))

XPCA<-prcomp(X, retx = TRUE)
library(relaimpo)
m10.PCA<-lm(Y~.,data=data.frame(Y=Y,XPCA$x))
metrics.PCA <- calc.relimp(m10.PCA, type = c("first"))
first.PCA.rank<-metrics.PCA@first.rank
orderedFactors<-XPCA$x[,order(first.PCA.rank)]
rSquaredOrderedPCA<-sapply(1:N.orig,function(z) summary(lm(Y~.,data=data.frame(Y=Y,orderedFactors[,1:z])))$r.squared)
N.PCA<-min(which(rSquaredOrderedPCA > 0.9))

mdr<-N.orig-N.PCA                                                                                   # Model dimensionality reduction
BestRsquared <- summary(lm(Y~.,data=data.frame(Y=Y,orderedFactors[,1:N.PCA])))$r.squared            # Determination coefficient
