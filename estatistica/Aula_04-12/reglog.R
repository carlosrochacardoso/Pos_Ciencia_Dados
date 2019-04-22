
dados <- read.csv(file.choose(), sep = ";", header = T)

summary(dados)

plot(dados$DESPESAS, dados$SITUACAO)

cor.test(dados$DESPESAS, dados$SITUACAO)

mod1 <- glm(SITUACAO ~ DESPESAS, data = dados, family = binomial)

summary(mod1)

plot(dados$DESPESAS, dados$SITUACAO, col='red', pch=20)
points(dados$DESPESAS, mod1$fitted.values, pch=4)

W <- table(dados$SITUACAO, predict(mod1)>0.5)
acuracia <- sum(diag(W)/sum(W))
precisao <- W[4]/sum(W[4], W[3])
revoc.recall <- W[4]/sum(W[4],W[2])
F1Scor <- 2*revoc.recall*precisao/sum(precisao,revoc.recall) 

install.packages("InformationValue")
library(InformationValue)

ks_stat(actuals = dados$SITUACAO, predictedScores = mod1$linear.predictors)

ks_plot(actuals = dados$SITUACAO, predictedScores = mod1$linear.predictors)  

plotROC(actuals = dados$SITUACAO, predictedScores = mod1$linear.predictors)  


install.packages("descr")
library(descr)

freq(dados$SITUACAO)  

prev.dados <- read.csv(file.choose(), sep = ";", header = T)

prev.dados$Prob <- predict(mod1, newdata = prev.dados, type = "response")

  