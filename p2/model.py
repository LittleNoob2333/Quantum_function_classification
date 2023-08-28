from deepquantum import *

class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n 
        self.cir = QubitCircuit(self.n+1)
        self.circuit()

    def circuit(self):
        for i in range(2):
            self.cir.rzlayer()
            self.cir.rylayer()
            self.cir.rzlayer()
            self.cir.cnot_ring()
            self.cir.barrier()
        self.cir.rzlayer()
        self.cir.rylayer()
        self.cir.rzlayer()
        for i in range(3):
            self.cir.observable(i)
    
    def forward(self, x):
        state = self.cir.amplitude_encoding(x)
        self.cir(state=state)
        exp = self.cir.expectation()
        return exp
    

    def get_results(self, x):
        """
        Args:
            x (tensor): a batch of black box funcions encoded as quantum states, shape=(batch_size, 2**(n+1), 2**(n+1))
        Return:
            predictions (list): a batch of classificaitons 
            of black box functions, e.g. [1, 2, 2, 0, 1, 1, 0, ...]
        """
        out = self(x)
        _, predictions = torch.max(out, 1) 
        return predictions