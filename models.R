rm(list = objects())

library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
library(ranger)
library(keras)
install_keras()
source("score.R")

data0 <- read_delim("data/Data0.csv", delim = ",")
data1 <- read_delim("data/Data1.csv", delim = ",")

range(Data1$Date)
range(Data0$Date)

data0$Time <- as.numeric(Data0$Date)
data1$Time <- as.numeric(Data1$Date)

sel_a <- which(data0$Year <= 2021)
sel_b <- which(data0$Year > 2021)