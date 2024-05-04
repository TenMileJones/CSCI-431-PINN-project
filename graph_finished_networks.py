import matplotlib.pyplot as plt
import numpy as np
import keras
import sys

# Author: Branson Jones
# Version: 5/2/24

# graph_finished_networks.py will take a saved neural network file 
# (ideally a .keras file) and graph it against a sin wave.

# Usage: python graph_finished_networks.py <nn_file_path>

if __name__ == "__main__":
    model_name = sys.argv[1]
    print(model_name)
    
    # instantiate model
    model = keras.models.load_model(model_name)
    
    # (loss function doesn't matter if not training)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[])
    
    # Define test data (currently in wide format, narrow is +-4pi)
    test_set = np.linspace((np.pi)*-10, np.pi*10, 400, dtype = np.float32)
    
    # Calculate correct output for the sake of comparison
    ideal_output = np.array([np.sin(x) for x in test_set], dtype = np.float32)
    
    # Plot
    prediction = model.predict(test_set, verbose=0)
    net_out = np.squeeze(prediction)  # makes sure it's the dimensionality we want

    plt.plot(test_set, ideal_output) # blue: the real deal
    plt.plot(test_set, net_out, "r--") # red: neural net prediction
    plt.axis((test_set[0], test_set[-1], -2, 2))
    plt.show()

