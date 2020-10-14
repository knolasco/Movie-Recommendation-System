if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#---------------------------------------------------------------------------------------------^ above is creating the dataset

#--------------------------------------------Project analysis begins below!

#since I will be using machine learning, the first thing I want to do is split edx into training and test sets.
#i will take 20% of the data as the test, and 80 to train.
library(caret)

test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
test_set <- edx[test_index,]
train_set <- edx[-test_index,]

#similarly as the machine learning course, I want to make sure to not include movies 
#and users in the test set that are not in the training set

test_set <- test_set %>% semi_join(train_set, by = "movieId")
test_set <- test_set %>% semi_join(train_set, by = "userId")


#I want to know if there are any NA's for the ratings in my training set
sum(is.na(train_set$rating)) #there are no NA's. that's good!

#my idea is to use a few models and make an ensemble of them, 
#we'll compare the accuracies of all of them and use the best!

#since we will be graded based on RMSE, I will create a function for it

RMSE <- function(actual, predicted){
  sqrt(mean((actual - predicted)^2))
}

#our first model will include only mean of movie ratings and movie effects, b_i :=  rating - avg

mu <- mean(train_set$rating)

movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#now we make our prediction using mean of movie ratings and movie effects

pred_1 <- mu + test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  pull(b_i)

rmse1 <- RMSE(test_set$rating, pred_1)

#With this first simple model, we get an RMSE of 0.944! Now I will try adding on the user effects.
#User effects are, b_u := rating - avg - b_i

user_effects <- train_set %>%
  left_join(movie_effects, by = "movieId") %>% #we need to join the tables in order to have access to b_i
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

pred_2 <- mu + test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(bu_bi = b_u + b_i) %>%
  pull(bu_bi)

rmse2 <- RMSE(test_set$rating, pred_2)

#By including the user effects into our model, we have a RMSE of 0.866! 
#Now I would like to regularize the movie effects and see if the RMSE improves

#the regularized movie effects equal, b_i := sum(rating - mu)/(n + lambda)
#I am not sure what lambda should be, so I will use cross validation to find the optimal value.

lambdas <- seq(0,10,0.1)

sums <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), nums = n())

rmse_cv <- sapply(lambdas, function(lam){#this function makes predictions using different values of lambda
  predictions <- test_set %>%
    left_join(sums, by = "movieId") %>%
    mutate(b_i = s/(nums + lam)) %>%
    mutate(preds = mu + b_i) %>%
    pull(preds)
  return(RMSE(test_set$rating, predictions))
})


lambdas[which.min(rmse_cv)]#this tells us that the optimal lambda to use is 1.7
#next we will create the model using lambda = 1.7

lambda <- 1.7 

reg_movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarize(reg_bi = sum(rating - mu)/(n() + lambda))

pred_3 <- test_set %>%
  left_join(reg_movie_effects, by = "movieId") %>%
  mutate(preds = mu + reg_bi) %>%
  pull(preds)

rmse3 <- RMSE(test_set$rating, pred_3) #the RMSE using this method is 0.944. So we didn't improve as much as I hoped.
#I will try regularizing by the user effect to see if there is a difference.

#the regularized user effects equal, b_u := sum(rating - mu - reg_bi)/(n + lambda)
#I am not sure what lambda should be, so I will use cross validation to find the optimal value.


rmses_cv2 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
    left_join(reg_user, by = "userId") %>%
    mutate(preds = mu + bi + bu) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})

lambdas[which.min(rmses_cv2)]
#the lambda that minimizes rmse for reg_user + reg_movie is 4.7

lambda2 <- 4.7

#now we run the model with lambda = 4.7 to get the rmse.

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda2 + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda2 + n()))

pred_4 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
  left_join(reg_user, by = "userId") %>%
  mutate(preds = mu + bi + bu) %>%
  pull(preds)

rmse4 <- RMSE(test_set$rating, pred_4)
#This is the lowest RMSE yet: 0.8655424.


#Now I will attempt to make an ensemble of the 4 models that were used in this analysis
#The idea is to average the ratings from the 4 models and use that as a prediction


pred_5 <- (pred_1 + pred_2 + pred_3 + pred_4)/4

rmse5 <- RMSE(test_set$rating, pred_5)
#this gave an rmse of 0.8847, which is more than the previous rmse. Perhaps I just average the two best methods?

RMSE(test_set$rating, (pred_2 + pred_4)/2)
#this gave an rmse of 0.8656878, which is better than the previous ensemble 
#but not that great. Maybe there is genre_effects
#so far regularized effects have worked best, so I will try regularized genre_effects
#reg_gen := sum(rating - mu - bi - bu)/(lambda + n)


rmses_cv3 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  reg_genre <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - bi - bu)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    mutate(preds = mu + bi + bu + b_g) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})



lambdas[which.min(rmses_cv3)]
#the lambda that minimizes rmse for reg_user + reg_movie + reg_genres is 4.5

lambda3 <- 4.5
#now we run the model with the optimal lambda

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda3 + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda3 + n()))

reg_genre <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda3 + n()))

pred_6 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  mutate(preds = mu + bi + bu + b_g) %>%
  pull(preds)

rmse6 <- RMSE(test_set$rating, pred_6)
#this gave a lower rmse of 0.8652434 but still not exactly what I want. 
#next is to see if there is a way to regulize it by year

#first, I will create a new column to represent the date for both test and train set

library(lubridate)
train_set <- train_set %>%
  mutate(date = as_datetime(timestamp), year = year(date), month = month(date)) 

test_set <- test_set %>%
  mutate(date = as_datetime(timestamp), year = year(date), month = month(date))
#now the train and test sets have a year and month. I will first model with year effects. then by month effect.

#first we use cross validation to find the best lambda for the lowest RMSE
lambdas <- seq(0,10,0.25)
rmses_cv4 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  reg_genre <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - bi - bu)/(lam + n()))
  
  reg_year <- train_set %>% 
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% 
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(reg_year, by = "year") %>%
    mutate(preds = mu + bi + bu + b_g + b_y) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})

lambda <- lambdas[which.min(rmses_cv4)] #The optimal lambda for this model is 4.5

#now we run the model with the optimal lambda and see what the rmse is

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda + n()))

reg_genre <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda + n()))

reg_year <- train_set %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lambda + n()))

pred7 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% 
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  mutate(preds = mu + bi + bu + b_g + b_y) %>%
  pull(preds)

rmse7 <- RMSE(test_set$rating, pred7)
#this gives an rmse of 0.8651899, which is not small enough.now we try year and month

rmses_cv5 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  reg_genre <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - bi - bu)/(lam + n()))
  
  reg_year <- train_set %>% 
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lam + n()))
  
  reg_month <- train_set %>% 
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(reg_year, by = "year") %>%
    group_by(month) %>%
    summarize(b_m = sum(rating - mu - bi - bu - b_g - b_y)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% 
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(reg_year, by = "year") %>%
    left_join(reg_month, by = "month") %>%
    mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})

lambda <- lambdas[which.min(rmses_cv5)] #this lambda is also 4.5, a little suspicious but we're going with it

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda + n()))

reg_genre <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda + n()))

reg_year <- train_set %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lambda + n()))

reg_month <- train_set %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  group_by(month) %>%
  summarize(b_m = sum(rating - mu - bi - bu - b_g - b_y)/(lambda + n()))

pred8 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% 
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  left_join(reg_month, by = "month") %>%
  mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
  pull(preds)

rmse8 <- RMSE(test_set$rating, pred8)
#this gave a lower rmse, but still not low enough.


methods <- c("Movie Effects",
             "Movie + User Effects",
             "Reg Movie Effects",
             "Reg Movie/User Effects",
             "Ensemble of predictions 1,2,3,4(just the mean)",
             "Reg Movie/User/Genres Effects",
             "Reg Movie/User/Genres/Year Effects",
             "Reg Movie/User/Genres/Year/Month Effects")
rmse_table <- c(rmse1,
                rmse2,
                rmse3,
                rmse4,
                rmse5,
                rmse6,
                rmse7,
                rmse8)

pred_table <- c("pred_1",
                "pred_2",
                "pred_3",
                "pred_4",
                "pred_5",
                "pred_6",
                "pred7",
                "pred8")


results_so_far <- data.frame(method = methods, rmse = rmse_table, pred_index = pred_table)
results_so_far %>% arrange(rmse)
#so far, the best method is regularized movie/user/genre/year/month effects.  
#let's try an ensemble of the regularized models

pred_9 <- (pred_6 + pred_4 + pred8 + pred7)/4
rmse9 <- RMSE(test_set$rating, pred_9)

#ok, now I will try to run a model with reg movie/user/genres effects and just normal year and month effects
#we will use cross validation to find the optimal lambda, for this model, lambda will not be used for year or month.


rmses_cv6 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  reg_genre <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - bi - bu)/(lam + n()))
  
  norm_year <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = mean(rating - mu - bi - bu - b_g))
  
  norm_month <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(norm_year, by = "year") %>%
    group_by(month) %>%
    summarize(b_m = mean(rating - mu - bi - bu - b_g - b_y))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(norm_year, by = "year") %>%
    left_join(norm_month, by = "month") %>%
    mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})


lambda <- lambdas[which.min(rmses_cv6)]

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda + n()))

reg_genre <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda + n()))

norm_year <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - bi - bu - b_g))

norm_month <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(norm_year, by = "year") %>%
  group_by(month) %>%
  summarize(b_m = mean(rating - mu - bi - bu - b_g - b_y))

pred_10 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(norm_year, by = "year") %>%
  left_join(norm_month, by = "month") %>%
  mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
  pull(preds)

rmse10 <- RMSE(test_set$rating, pred_10)


methods <- c("Movie Effects",
             "Movie + User Effects",
             "Reg Movie Effects",
             "Reg Movie/User Effects",
             "Ensemble of predictions 1,2,3,4(just the mean)",
             "Reg Movie/User/Genres Effects",
             "Reg Movie/User/Genres/Year Effects",
             "Reg Movie/User/Genres/Year/Month Effects",
             "Reg Movie/User/Genres and norm Year/Month effects")
rmse_table <- c(rmse1,
                rmse2,
                rmse3,
                rmse4,
                rmse5,
                rmse6,
                rmse7,
                rmse8,
                rmse10)

pred_table <- c("pred_1",
                "pred_2",
                "pred_3",
                "pred_4",
                "pred_5",
                "pred_6",
                "pred7",
                "pred8",
                "pred10")


results_so_far <- data.frame(method = methods, rmse = rmse_table, pred_index = pred_table)
results_so_far %>% arrange(rmse)

#ok, so reg movie/user/genre/year/month is the best method. 
#I want to see whether I should be rounding or truncating the predictions

#first, I want to see the distributions of the values of the ratings in edx

rating_dist <- edx %>% select(rating)

rating_dist %>% 
  group_by(rating) %>% 
  ggplot(aes(x = rating)) +
  geom_bar()

#this shows that most ratings are integers but there are ratings in increments of 0.5
#I will now create a function that changes my predictions to increments of 0.5 depending on how close the prediction is.

round_rating <- function(rating){
  if(abs(rating - 5) < 0.25){
    return(5)
  }
  if(abs(rating - 4.5) < 0.25){
    return(4.5)
  }
  if(abs(rating - 4) < 0.25){
    return(4)
  }
  if(abs(rating - 3.5) < 0.25){
    return(3.5)
  }
  if(abs(rating - 3) < 0.25){
    return(3)
  }
  if(abs(rating - 2.5) < 0.25){
    return(2.5)
  }
  if(abs(rating - 2) < 0.25){
    return(2)
  }
  if(abs(rating - 1.5) < 0.25){
    return(1.5)
  }
  if(abs(rating - 1) < 0.25){
    return(1)
  }
  else{
    return(0.5)
    }
}

#i want to make sure this function works the way I want it to, so i will test with a random vector
test_vec <- c(0.11,5.2,1.1,2.456,3.11,3.5,3.54)
sapply(test_vec, round_rating)

#ok so the function works. so I will apply it to pred8

pred_8rounded <- sapply(pred8, round_rating)

RMSE(test_set$rating, pred_8rounded)

#this didn't give good results. So instead of rounding closer to the 0.5 increments, we'll just round to the nearest 10th.

pred_8rounded <- round(pred8, 1)

RMSE(test_set$rating, pred_8rounded)

#this didnt improve the rmse, so instead I will truncate.

pred_8trunc <- as.integer(pred8*10)/10

RMSE(test_set$rating, pred_8trunc)

#this also didn't improve, so we will keep the model as is.


#ok so the best method is all regularized features.
#so to get ready to have the final rmse, we will add the year and month to edx and validation

edx <- edx %>%
  mutate(year = year(as_datetime(timestamp)), month = month(as_datetime(timestamp)))

validation <- validation %>%
  mutate(year = year(as_datetime(timestamp)), month = month(as_datetime(timestamp)))



lambda <- lambdas[which.min(rmses_cv5)] 

reg_movie <- edx %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda + n())) #reg_movie will contain bi

reg_user <- edx %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda + n()))

reg_genre <- edx %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda + n()))

reg_year <- edx %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lambda + n()))

reg_month <- edx %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  group_by(month) %>%
  summarize(b_m = sum(rating - mu - bi - bu - b_g - b_y)/(lambda + n()))

pred_final <- validation %>%
  left_join(reg_movie, by = "movieId") %>% 
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  left_join(reg_month, by = "month") %>%
  mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
  pull(preds)

final_rmse <- RMSE(validation$rating, pred_final)
final_rmse
#pred_final contains the final predictions using the validation set.
