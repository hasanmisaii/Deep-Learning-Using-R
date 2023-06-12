# Recurrent neural network 
# Change your environment to the python environment
reticulate::use_condaenv("python.exe")
# Install keras and tensorflow in the R terminal using pip and install.packages()
library(keras)

# Split the data into training and test sets
df <- data
split <- floor(0.70 * nrow(df))
train_data <- df[1:split, ]
test_data <- df[(split + 1):nrow(df), ]

## scale and normalize data
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}

scaled <- scale_data(train_data[, -ncol(df)], test_data[, -ncol(df)], c(-1, 1))
# Although, time ranges in [0, \infty), but I scaled it to (-1, 1) to consider non-momotonic increasing trend.

# Normalize the data
#scaler <- caret::preProcess(as.data.frame(train_data[, -ncol(df)]), method = c("center", "scale"))
#train_data[, -ncol(df)] <- predict(scaler, as.data.frame(train_data[, -ncol(df)]))
# test_data[, -ncol(df)]  <- predict(scaler, as.data.frame(test_data[, -ncol(df)]))
train_data[, -ncol(df)] <- scaled$scaled_train
test_data[, -ncol(df)] <- scaled$scaled_test

######AR(q), q = ?    #If the data follows AR models one way for look-back number is finding the level of AR, but it doesn't work allways. 
# Load a time series data
# data <- df$V2
# Fit AR models with order ranging from 1 to 10
# order_max <- 10
# fit <- vector("list", order_max)

# for (i in 1:order_max) {
#  fit[[i]] <- ar(data, order.max=i, method="yule-walker")
#}

# Select the order with the lowest AIC
# aic <- sapply(fit, function(x) x$aic)
# order_aic <- which.min(aic[[10]])

# Print the selected order
# cat("Order selected by AIC:", order_aic)


# Convert the data into a 3D array for input into the BiLSTM
create_dataset <- function(data, look_back = 1) {
  data_X_0 <- data_X_1 <- data_y <- c()
  for (i in 1:(nrow(data) - look_back)) {
    a <- data[(i + look_back):(i + look_back), -ncol(data)]
    b <- data[i:(i + look_back-1), ncol(data)]
    data_X_0 <- rbind(data_X_0, a)
    data_X_1 <- rbind(data_X_1, b)
    data_y <- c(data_y, data[i + look_back, ncol(data)])
  }
  list(data_X = array(cbind(data_X_0, data_X_1), dim = c(nrow(data_X_1), look_back + num_features , ncol(data) - 1)), data_y = array(data_y))
}

look_back <- 2; num_features = 1
train_dataset <- create_dataset(train_data, look_back)
test_dataset <- create_dataset(test_data, look_back)



## Define BiLSTM Model
model <- keras_model_sequential()
model %>%
   bidirectional(layer_lstm(units, return_sequences = FALSE, go_backwards = FALSE),
                   backward_layer = layer_lstm(units, return_sequences = FALSE, go_backwards = TRUE),
                   merge_mode = "concat",
                   input_shape = c(time_steps = look_back+1, ncol(train_data)-1)) %>%
   # layer_dropout(rate = 0.5) %>%      #regularization to prevent overfitting 
   layer_dense(units = 1) 


# First visualization of the model
print(model)

# Plot the model
library(deepviz)
library(magrittr)
#library(kerasR)

# plot the model
model %>% 
  plot_model()

# Define metrics, loss and optimizer functions 
# Define R-squared metric function
rsq_metric <- function(y_true, y_pred) {
              ss_res <- k_sum(k_square(y_true - y_pred))
              ss_tot <- k_sum(k_square(y_true - k_mean(y_true)))
              return(1 - ss_res/(ss_tot + k_epsilon()))
}

model %>% compile(
        loss = 'MSE',
        optimizer = optimizer_adam(learning_rate= 0.001),  
        metrics = list('MAE', rsq_metric)
)



# Train the model
history <- model %>% fit(
                train_dataset$data_X, train_dataset$data_y,
                epochs = 100, batch_size = 28, verbose = 2, 
                validation_split = 0.2, 
                shuffle = FALSE
)


plot(history)

# Evaluate the model on the test set
mse <- model %>%
             evaluate(test_dataset$data_X, test_dataset$data_y, verbose = 0)
             
cat("Mse & MAE on Test Set:", mse)

testset_prediction <- model %>%
                             predict(test_dataset$data_X)


trainset_prediction <- model %>%
                             predict(train_dataset$data_X)
