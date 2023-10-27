# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Function to create a CNN model
def create_cnn_model(input_shape, kernel_size, pool_size, drop):
    model = Sequential()

    # Add the first convolutional layer with 32 filters, ReLU activation, and input shape
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    
    # Add a second convolutional layer with 64 filters and ReLU activation
    model.add(Conv2D(64, kernel_size, activation='relu'))
    
    # Add a max-pooling layer
    model.add(MaxPool2D(pool_size=pool_size))
    
    # Add dropout to prevent overfitting
    model.add(Dropout(drop))
    
    # Flatten the output for fully connected layers
    model.add(Flatten())
    
    # Add a fully connected layer with 256 neurons and ReLU activation
    model.add(Dense(256, activation='relu'))
    
    # Add dropout again
    model.add(Dropout(drop))
    
    # Add the output layer with 10 neurons for classification (softmax activation)
    model.add(Dense(10, activation='softmax'))

    # Compile the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Function for training the model with early stopping
def train_model(cnn_model, X_train, y_train, X_test, y_test, epochs):
    # Set up early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    
    # Train the model with the provided data and epochs, using early stopping
    cnn_model.fit(X_train, y_train, epochs=epochs, callbacks=[early_stop], validation_data=(X_test, y_test))
    
    return cnn_model

# Function to store the model as JSON and its weights as HDF5
def store_model(model, file_path='../output/', file_name='trained_model'):
    # Serialize model architecture to JSON
    model_json = model.to_json()
    with open(file_path + file_name + '.json', "w") as json_file:
        json_file.write(model_json)
    
    # Serialize model weights to HDF5
    model.save_weights(file_path + file_name + '.h5')
    
    # Print confirmation messages
    print(f"Saved model to disk in path {file_path} as {file_name + '.json'}")
    print(f"Saved weights to disk in path {file_path} as {file_name + '.h5'}")
