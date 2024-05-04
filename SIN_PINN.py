
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

# Author: Branson Jones
# Version: 5/3/24

# SIN_PINN.py tries to train a PINN that approximates a sin wave and saves it before 
# graphing the network against a sin wave for comparisons sake. Training is done stochastically.
# Hyperparameters are easily modifiable to create different training architechtures. 
# Boundary conditions are also straightforward to modify, add, and remove.

# Read the README for information about the PINNs saved in this repo.

def create_nn_model(num_hidden_layers, num_neurons_per_layer):
    model = keras.Sequential()
    
    # Input is 1-dimensional (just x-values)
    # Must be tuple.
    model.add(keras.Input((1,)))
    
    # append hidden layers
    for _ in range(num_hidden_layers):
        model.add(keras.layers.Dense(num_neurons_per_layer, 
            activation='tanh', kernel_initializer='glorot_normal'))
        
        model.add(keras.layers.Dropout(0.5))
        
        # output doesn't graph with batch normalization and I don't know why.
        # Can't turn it to training mode? :/
        # model.add(keras.layers.BatchNormalization())
    
    # Output is 1-dimensional (just y-values)
    model.add(keras.layers.Dense(1))
    
    return model

# Residual is how far off the model is from what we want
def calc_residual(model, x_value):
    
    with tf.GradientTape(persistent=True) as tape:
        # Watch x_value to compute derivative
        tape.watch(x_value)
        
        # Determine f(x_value)
        f = model(x_value)
        
        # Compute first derivative f'(x_value) within gradient tape
        f_x = tape.gradient(f, x_value)
    
    # Compute second derivative f''(x_value)
    f_xx = tape.gradient(f_x, x_value)
    
    # Delete tape
    del tape
    
    # Return (squared) residual
    return (f + f_xx)**2

# Loss is determined by residual and boundary conditions
def calc_loss(model, x_value, twoD_boundary_in, boundary_out):
    # boundary_in and boundary_out should be same length
    
    # Compute (squared) residual
    r = calc_residual(model, x_value)
    
    # Initialize loss
    loss = r
    
    # Boundary condition loss: how far off model is from
    # being where we want at fixed points
    for i in range(len(twoD_boundary_in)):
        # add deviation from each boundary to loss
        loss += (model(twoD_boundary_in[i])-boundary_out[i])**2
    
    return loss
    
    
if __name__ == "__main__":
    
    print("Don't forget to change your network file name! Don't overwrite your progress.")
    
    # Set data type
    DTYPE='float32'
    keras.backend.set_floatx(DTYPE)
    
    # Set boundary conditions. f(boundary_in[i]) = boundary_out[i]
    boundary_in = [(np.pi)*6.5, -6.5*(np.pi), (np.pi)*7.5, (np.pi)*-7.5, (np.pi)*8.5, -8.5*(np.pi)]
    boundary_out = [1, -1, -1, 1, 1, -1]
    
    # Create 2D tensor version of boundary_in
    boundary_in_as_tensor = tf.convert_to_tensor(boundary_in)
    twoD_boundary_in = tf.reshape(boundary_in_as_tensor, (-1, 1))
    
    # Define training data
    training_set = np.linspace((np.pi)*-10, np.pi*10, 400, dtype = np.float32)
    # Convert to tensor and add dimension so tensorflow gets the 2D input it expects
    training_set_as_tensor = tf.convert_to_tensor(training_set, tf.float32)
    twoD_training_set = tf.reshape(training_set_as_tensor, (-1, 1))
    
    # Define model structure
    model = create_nn_model(num_hidden_layers = 4, num_neurons_per_layer = 20)
    
    # Define parallel training strategy
    # communication_options = tf.distribute.experimental.CommunicationOptions(
    #     implementation=tf.distribute.experimental.CommunicationImplementation.RING)
    # strategy = tf.distribute.MultiWorkerMirroredStrategy(
    #     communication_options=communication_options)
    
    # The custom training loop is only using one CPU core compared to standard TF which 
    # likes to use all of them. I wanted to parallelize but its a bit too in the weeds
    # for this short of a turnaround. I wonder if it would have to be parallelized to be
    # faster on a GPU, not sure how 'cores' work in that context. Is a GPU treated as one core?
    
    # Define hyperparameters
    epochs = 2000
    n_steps = len(twoD_training_set)
    optimizer = keras.optimizers.Adam()
    
    # Train
    for epoch in range(1, epochs + 1):
        
        # print epoch number
        print("Epoch {}/{}".format(epoch, epochs))
        
        for step in range(n_steps):
            x_in = twoD_training_set[step]
            
            with tf.GradientTape() as tape:
                
                # Find prediction for this x
                y_pred = model(x_in, training=True)
                
                # Calculate loss associated with this x_in y_pred pair (it's calculated again in calc_loss)
                loss = calc_loss(model, x_in, twoD_boundary_in, boundary_out)
            
            # Find gradient of loss and apply it to model with the optimizer
            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            
    
    # Save
    model.save("2000_wide_attempt.keras", include_optimizer = False)
    
    # Create test set
    test_set = training_set # Currently just training set - let's get it to run first
    
    # Calculate correct output for the sake of comparison
    ideal_output = np.array([np.sin(x) for x in test_set], dtype = np.float32)
    
    # Plot
    prediction = model.predict(test_set, verbose=0)
    net_out = np.squeeze(prediction)  # makes sure it's the dimensionality we want

    plt.plot(test_set, ideal_output) # blue: the real deal
    plt.plot(test_set, net_out, "r--") # red: neural net prediction
    plt.axis((test_set[0], test_set[-1], -2, 2))
    plt.show()
    
    
    
