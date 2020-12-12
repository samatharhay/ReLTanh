import torch.nn as nn

from test_function import ReLTanh


## Define the NN architecture
class FCNN(nn.Module):
    def __init__(self, args, input_dims=3 *32 *32, output_dims=10,
                 hidden_dims=128, hidden_layers=5):
        super(FCNN, self).__init__()
        self.dataset = args.dataset
        # construct layers
        in_dims = input_dims
        self.layers = []
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(in_dims, hidden_dims))
            self.layers.append(self.act_fn_layer(args))
            in_dims = hidden_dims
        self.model = nn.Sequential(*self.layers)
        self.out = nn.Linear(in_dims, output_dims)

        # activation for evaluation
        self.act = nn.Softmax(dim=output_dims)

    def act_fn_layer(self, args):
        if args.activation_function_type == 'reltanh':
            return ReLTanh(args.reltanh_alpha, args.reltanh_beta)
        elif args.activation_function_type == 'relu':
            return nn.ReLU()
        elif args.activation_function_type == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError(f'Did not implement activation function: {act_fn_name}')

    def forward(self, x):
        if self.dataset == "cifar10":
            x = x.view(-1, 3 * 32 * 32)  # [bn, 28, 28] -> [bn, 768]
        elif self.dataset == "mnist":
            x = x.view(-1, 28*28)
        x = self.model(x)        # [bn, 768] -> [bn, 128]
        x = self.out(x)          # [bn, 128] -> [bn, 10]
        return x

    def classify(self, x):
        x = self(x)
        x = self.act(x)
        return torch.argmax(x, dim=1)


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, args, in_channels=1, output_dims=10,
                 hidden_dims=16, hidden_layers=5):
        super(CNN, self).__init__()

        if hidden_layers <= 3:
            raise NotImplementedError("CNN layers must be 4 or more")
        hidden_layers -= 3

        # construct layers
        # if args.dataset == 'MNIST':
        in_dims = in_channels
        self.layers = []

        # these first 3 layers downsample into 4x4
        for _ in range(3):
            out_dims = in_dims * 2
            self.layers.append(nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=3//2, stride=2))
            self.layers.append(self.act_fn_layer(args))
            in_dims = out_dims

        # hidden layers
        for _ in range(hidden_layers):
            out_dims = hidden_dims
            self.layers.append(nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=3//2))
            self.layers.append(self.act_fn_layer(args))
            in_dims = out_dims

        # define model
        self.model = nn.Sequential(*self.layers)
        self.feature_dims = in_dims * 4 * 4
        self.out = nn.Linear(self.feature_dims, output_dims)

        # activation for evaluation
        self.act = nn.Softmax(dim=output_dims)

    def act_fn_layer(self, args):
        if args.activation_function_type == 'reltanh':
            return ReLTanh(args.reltanh_alpha, args.reltanh_beta)
        elif args.activation_function_type == 'relu':
            return nn.ReLU()
        elif args.activation_function_type == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError(f'Did not implement activation function: {act_fn_name}')

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.feature_dims)
        x = self.out(x)
        return x

    def classify(self, x):
        x = self(x)
        x = self.act(x)
        return torch.argmax(x, dim=1)
