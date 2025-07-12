import numpy as np

class Layer_SimpleRNN:
    def __init__(self):
        self.timesteps = 100
        self.inputs = 10
        self.output = 64

    def forward(self):
        self.input_matrix  = np.random.random((self.timesteps,self.inputs))
        self.matrix = np.zeros((self.output,))

        self.w = np.random.random((self.output,self.inputs))
        self.u = np.random.random((self.output, self.output))
        self.b  = np.random.random((self.output,))

        self.successive = []

        for self.input_t in self.input_matrix:
           self.output_t = np.tanh(np.dot(self.w,self.input_t) + np.dot(self.u,self.matrix) + self.b)
           self.successive.append(self.output_t)
           self.matrix = self.output_t

        self.final_output = np.concatenate(self.successive,axis = 0)
        print(self.final_output)

if __name__ == '__main__' : 
    start = Layer_SimpleRNN()
    start.forward() 
        