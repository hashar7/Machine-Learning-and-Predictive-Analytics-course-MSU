library(glmnet)

data <- read.csv("test_sample.csv", sep = ",")
X<-data.matrix(data[,2:492])
Y<-data[,1]
set.seed(1) 

###################################################################
###################################################################

set.seed(1) 
cv.out=cv.glmnet(x=X,y=Y,alpha=1)
(bestlam =cv.out$lambda.min) # 0.1174731
out=glmnet(x=as.matrix(data[,2:492]),
           y=as.vector(data[,1]),alpha=1,lambda=bestlam)
lasso.coef=predict(out,type="coefficients",s=bestlam)
head(lasso.coef)
eliminatedByLasso = which(lasso.coef[-1] == 0, 
                          arr.ind = FALSE, useNames = FALSE)
length(eliminatedByLasso) # 60
linear_model = lm(Y~., data=data.frame(Y=Y, X=X))
indecses = coefficients(summary(linear_model))[-1,4] > 0.05
eliminatedByLm =  which(indecses, arr.ind = FALSE, useNames = TRUE)
length(eliminatedByLm) # 104

res = matrix(c("lasso","lm","",""),ncol=2)
colnames(res) <- c("model","removed_regressors")
res[,"removed_regressors"][1] = paste0(eliminatedByLasso,collapse = " ")
res[,"removed_regressors"][2] = paste0(eliminatedByLm,collapse = " ")
write.csv(res,"W2answer.csv",quote=FALSE,row.names = F)
