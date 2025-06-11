import numpy as np
import pickle

class NeuralNetwork: 
    def __init__(self): 
        self.no_of_inputs = 28 * 28
        self.no_of_hidden_layers = 2
        self.hidden_layer_neuron = [512, 512]
        self.no_of_outputs = 10
        self.store_output = []
        self.weights = []
        self.biases = []

    def initialise(self):
        self.weights.append(
            np.random.randn(self.no_of_inputs, self.hidden_layer_neuron[0]) * np.sqrt(2 / self.no_of_inputs)
        )
        self.biases.append(np.zeros(self.hidden_layer_neuron[0]))
        
        for j in range(self.no_of_hidden_layers - 1):
            self.weights.append(
                np.random.randn(self.hidden_layer_neuron[j], self.hidden_layer_neuron[j + 1]) * np.sqrt(2 / self.hidden_layer_neuron[j])
            )
            self.biases.append(np.zeros(self.hidden_layer_neuron[j + 1]))
        
        self.weights.append(
            np.random.randn(self.hidden_layer_neuron[-1], self.no_of_outputs) * np.sqrt(2 / self.hidden_layer_neuron[-1])
        )
        self.biases.append(np.zeros(self.no_of_outputs))

    def Forward_prop(self, inputs): 
        inputs = np.array(inputs / 255)
        self.store_output = [] 
        output = self.activation_func(np.add(np.dot(inputs, self.weights[0]), self.biases[0]))  # checked
        self.store_output.append(inputs)
        self.store_output.append(output)
        
        for j in range(1, len(self.weights)):
            output = self.activation_func(np.add(np.dot(output, self.weights[j]), self.biases[j]))
            self.store_output.append(output)
        return output

    def activation_func(self, output): 
        return np.where(output > 0, output, 0.01 * output)  

    def soft_loss(self, output): 
        exp_output = np.exp(output - np.max(output)) 
        sum_exp = np.sum(exp_output)
        return exp_output / sum_exp

    def entropy_loss(self, output): 
        return -np.log(output)

    def back_propagation(self, actual_output, soft_loss): 
        weight_store_derivative = []
        biases_store_derivative = []
        last_activation_derivatives = np.subtract(actual_output, soft_loss)  # checked
        activational_derivatives_store = [last_activation_derivatives]
        # A(n-1) = w11(A(n)) + w12(B(n)) ... so onnnnn 
        # so we can say activation(n)*transpose(Weights) === (1*10)*(10*512) makes sense
        for j in range(self.no_of_hidden_layers, 0, -1): 
            activational_derivatives_store.append(
                np.dot(activational_derivatives_store[-1], np.transpose(self.weights[j]))
            )
        activational_derivatives_store = activational_derivatives_store[::-1]  # reverse
        #before all this its better to calc all the activational deriatives before weights and biases
        # we have weights between 512 x 10 so W(axb) = derivat(ive(last_activation[b])*previous_activation[a]*(derivative of the loss function wrt activation of A(n)[b]))
        # we have B[a] = no previous activation like in weights 
        something = np.vectorize(self.derivative) # derivative of activation function
        # weights_change = np.dot(np.transpose(An_1) , np.multiply(something(An),lastactivation_derivatives))
        # biases_change = np.multiply(something(An),lastactivation_derivatives)
        for j in range(len(self.weights)):
            # print(self.store_output[j].shape , np.matrix.transpose(np.multiply(something(self.store_output[j+1]),activational_derivatives_store[j])).shape) 
            a = self.store_output[j]
            b = np.multiply(something(self.store_output[j + 1]), activational_derivatives_store[j])
            a = a.reshape(-1, 1)
            b = b.reshape(-1, 1)
            weight_store_derivative.append(np.dot(a, np.transpose(b)))
            biases_store_derivative.append(b.flatten())
        
        return weight_store_derivative, biases_store_derivative

    def derivative(self, value): 
        return 0.01 if value <= 0 else 1  # Leaky ReLU derivative

    def learning(self, list_of_inputs, list_of_solutions, learning_rate):
        self.initialise()   
        accuracy = 0.1
        weights_change_sum = [np.zeros_like(w) for w in self.weights]
        biases_change_sum = [np.zeros_like(b) for b in self.biases]
        counter = 0
        for i in range(8000): 
            p = np.zeros(10)
            p[list_of_solutions[i]] = 1
            final_output = self.Forward_prop(list_of_inputs[i].flatten())
            soft_loss_prob = self.soft_loss(final_output)
            accuracy +=  np.sum(np.multiply(soft_loss_prob, p))
            weights_change, biases_change = self.back_propagation(p, soft_loss_prob)
            counter += 1 
            weights_change_sum = [np.add(weights_change_sum[j],weights_change[j]) for j in range(len(self.weights))]
            biases_change_sum = [np.add(biases_change_sum[j],biases_change[j]) for j in range(len(self.biases))]
            if counter == 1: 
                for j in range(len(self.weights)): 
                    
                    self.weights[j] += (learning_rate / 1) * weights_change_sum[j]
                    self.biases[j] += (learning_rate / 1) * biases_change_sum[j]
                counter = 0
                weights_change_sum = [np.zeros_like(w) for w in self.weights]
                biases_change_sum = [np.zeros_like(b) for b in self.biases]
                print(f"The no of iterations completed: {i + 1}, Accuracy: {accuracy/(i+1)}")
    def exam(self,images, solutions, start , length):
        ans = 0
        for i in range(length):  
            final_output = self.Forward_prop(images[i + start].flatten())  
            soft_loss_prob = self.soft_loss(final_output)  
            if np.argmax(soft_loss_prob) == solutions[i + start]: 
                ans += 1
        accuracy = ans / length 
        print(accuracy)
    