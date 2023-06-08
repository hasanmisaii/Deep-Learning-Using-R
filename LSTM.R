# Change your environment to the python environment
reticulate::use_condaenv("python.exe")


# Install keras and tensorflow in the R terminal using pip and install.packages()
library(keras)

model <- keras_model_sequential()
model %>%
   layer_lstm(units, return_sequences = FALSE, input_shape = c(num_features, ncol(train_data)-1), activation = 'relu') %>%
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

