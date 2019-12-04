#Matrix mathematics library
import numpy
#Library for the sigmoid function expit()
import scipy.special
#Library to load in png images
import imageio
#Library to search for wildcard files
import glob

#Basic Neural Network class
class neuralNetwork:

    #Initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #Set the nodes in the neural network
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #Create the initial weights between each layer
        self.weight_ih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.weight_ho = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        #How fast the neural network learns
        self.lr = learningrate

        #Sigmoid function for error correction
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #Train the neural network
    def train(self, epochs, dataset):
        for e in range(epochs):
            for data in dataset:
                #Create the target output default values
                targetset = numpy.zeros(self.onodes) + 0.01

                #Associate the target label to its nodes
                if data[0] == "A":
                    targetset[0] = 0.99
                elif data[0] == "B":
                    targetset[1] = 0.99
                elif data[0] == "C":
                    targetset[2] = 0.99
                elif data[0] == "D":
                    targetset[3] = 0.99
                elif data[0] == "E":
                    targetset[4] = 0.99
                elif data[0] == "F":
                    targetset[5] = 0.99

                #Convert the dataset to a 2d array
                inputs = numpy.array(data[1:], ndmin=2).T
                #Convert the 2d array input to a float input
                inputs = inputs.astype(float)
                #Convert the targetset label value to a 2d array
                targets = numpy.array(targetset, ndmin=2).T

                #Calculate the data going into and coming from the hidden layer
                hidden_outputs = self.activation_function(numpy.dot(self.weight_ih, inputs))

                #Calculate the data going into and coming from the output layer
                final_outputs = self.activation_function(numpy.dot(self.weight_ho, hidden_outputs))

                #Calculate the output layer error
                output_errors = targets - final_outputs
                #Calculate the hidden layer error
                hidden_errors = numpy.dot(self.weight_ho.T, output_errors)

                #Reweight the links between the input and hidden layers
                self.weight_ih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                      numpy.transpose(inputs))

                #Reweight the links between the hidden and output layers
                self.weight_ho += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                numpy.transpose(hidden_outputs))
                pass
            pass
        pass

    #Run data to test recognition
    def query(self, dataset):
        #Convert the dataset to a 2d array
        inputs = numpy.array(dataset, ndmin=2).T
        #Convert the 2d array input to a float input
        inputs = inputs.astype(float)

        #Calculate the data going into and coming from the hidden layer
        hidden_outputs = self.activation_function(numpy.dot(self.weight_ih, inputs))

        #Calculate the data going into and coming from the output layer
        final_outputs = self.activation_function(numpy.dot(self.weight_ho, hidden_outputs))

        return final_outputs

#Function to create, train, and test the neural network
def analyze():
    # load in training data
    training_dataset = loadImage('Training/Training_?.png')
    # load in the test data
    dataset = loadImage('Images/test_?.png')

    # Input node amount - amount of pixels in image
    input_nodes = len(dataset[0][1:])
    # hidden node amount
    hidden_nodes = 200
    # output node amount
    output_nodes = 6

    # Chance of the neural network to be correct
    learning_rate = 0.1

    # Create an instance of the neural network based on the inputted data
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # Train the neural network using the training images
    n.train(100, training_dataset)
    #Run recognition on all of the test images
    for x in range(len(dataset)):
        #Grab the label of the tested image
        correct_label = dataset[x][0]
        #Run through recognition to get output weights
        outputs = n.query(dataset[x][1:])

        #Save the highest output produced
        label = numpy.argmax(outputs)

        #Convert index back to actual label
        if label == 0:
            label = "A"
        elif label == 1:
            label = "B"
        elif label == 2:
            label = "C"
        elif label == 3:
            label = "D"
        elif label == 4:
            label = "E"
        elif label == 5:
            label = "F"

        #Display the highest output
        print("network says ", label)
        #Check to see if the calculated output is what the image is
        if label == correct_label:
            print("match!")
        else:
            print("no match!")
            pass

def loadImage(imagePath):

    #Array of reshaped images
    dataset = []

    # load the png image data as data set
    for image_file_name in glob.glob(imagePath):
        #Store the image label
        label = image_file_name[-5:-4]

        #State which image is being loaded
        print("loading ... ", image_file_name)
        img_array = imageio.imread(image_file_name, as_gray=True)

        #Flatten the image matrix to an array and invert
        img_data = 255.0 - img_array.reshape(784)

        #Condense the image data to a range of 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01

        #Combine the label and image data into one array of the dataset
        dataset.append(numpy.append(label, img_data))

    return dataset

#Introduction to program
print("Athena is a neural network that does image recognition.")
print("To train the neural network create multiple images labeled with what it is. (Letters A - F)")
print("The images need to be 28 x 28 and named Training_N.png. N being the label of the image.")
print("Put these multiple images in a folder named 'Training' next to Athena.py")
print("You can then test the accuracy of the image recognition by creating 28 x 28 images\nnamed test_N.png. N being the label of the image and then putting\nthem in a folder named 'Images' next to Athena.py\n")

analyze()
