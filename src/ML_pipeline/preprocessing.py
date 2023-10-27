from tensorflow.keras.utils import to_categorical

# Function to reshape data
def data_reshape(data, shape):
    # Reshaping data into a 4D shape if the length of the 'shape' array is 4
    if len(shape) == 4:
        reshape_data = data.reshape(shape[0], shape[1], shape[2], shape[3])  # Reshape the data
        return reshape_data
    else:
        print("The length of the reshape array is not 4")

# Function to convert numerical data to categorical
def convert_to_cat(data, labels):
    # Converting numerical data to categorical with 'labels' number of classes
    if labels != 0:  # The number of classes can't be 0
        data_cat = to_categorical(data, labels)  # Convert numerical data to categorical
        return data_cat
    else:
        print("The value for 'labels' cannot be 0")

# Function for feature scaling
def feature_scale(data, data_type, divisor):
    # Changing the numerical data type to float32
    if data_type == 'float32':
        data = data.astype(data_type)  # Convert the data type of the data to float32
    else:
        print("Type is not float32")

    if divisor != 0:
        data /= divisor  # Divide data by a non-zero divisor
        return data
    else:
        print("Division by zero is not allowed")
