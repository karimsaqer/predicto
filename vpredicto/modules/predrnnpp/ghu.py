'''
Gradient Highway Unit (GHU)
__init__ method to initialize the GHU: you can pass the input_channels, hidden_channels, and kernel_size as parameters
forward method to pass the input and hidden state through the GHU
the input is the input and hidden state
the output is the updated hidden state
'''
class GHU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(GHU, self).__init__()

        self.filter_size = kernel_size
        self.padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.z_gate = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.x_gate = nn.Conv2d(input_channels, hidden_channels * 2, kernel_size=kernel_size, padding=self.padding, bias=False)

    def forward(self, x, z):
        if z is None:
            z = torch.zeros_like(x)
        z_gate = self.z_gate(z)
        x_gate = self.x_gate(x)

        gates = x_gate + z_gate
        p, s = torch.split(gates, self.hidden_channels, dim=1)
        p = torch.tanh(p)
        s = torch.sigmoid(s)
        '''
        The equation z_new = s * p + (1-s) * z provides a mechanism to control how much
        of the new transformed information p is mixed with the old hidden state z.
        If s is close to 1, more of p is used; if s is close to 0, more of z is retained.
        '''
        z_new = s * p + (1-s) * z
        return z_new