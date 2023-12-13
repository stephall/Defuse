# Examples for either denoising or property models

# Import public modules
import torch
from importlib import reload

# Import custom modules
from src import encodings

# Reload custom modules
reload(encodings)

class DenoisingModel(torch.nn.Module):
    def __init__(self, x_enc_dim, t_dim=1, lat_dim=100):
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.x_enc_dim = x_enc_dim
        self.t_dim     = t_dim
        self.lat_dim   = lat_dim

        # Define the input and output dimension, based on the
        self.input_dim  = x_enc_dim + t_dim

        # Define an encoding for the x-values (i.e. categoricals)
        spatial_encoding  = encodings.OneHotEncoding1D(dim=self.x_enc_dim)
        self.encode_space = lambda x: spatial_encoding(x)

        # Define an encoding for the times (passed as scalars)
        time_encoding    = encodings.MultiLinearEncoding1D(dim=self.t_dim)
        self.encode_time = lambda t: time_encoding(t)

        # Define the model parts
        self.linear_1 = torch.nn.Linear(self.input_dim, self.lat_dim)
        self.linear_2 = torch.nn.Linear(self.lat_dim, self.lat_dim)
        self.linear_3 = torch.nn.Linear(self.lat_dim, self.x_enc_dim)

        # Define an activation function
        self.activation_fn = torch.nn.ReLU()

        # Define the softmax function that should be applied along the
        # second axis (i.e. the feature axis)
        self.softmax_fn = torch.nn.Softmax(dim=1)

    def forward(self, data_dict, t):
        """
        Define forward pass of the model. 
        
        Args:
            data_dict (dict): Dictionary holding the data as torch tensors.
                Required dictionary-key is 'x' and potential dictionary-key
                is 'y'.
            t (torch.tensor): Time as torch tensor.
        
        """
        # Access the data
        x = data_dict['x']

        # print("In forward of 'model'.")
        # print(f"x.shape: {x.shape}")
        # print(f"x[:10]: {x[:10]}")

        # Encode the spatial features of the data (i.e. x)
        x_encoded = self.encode_space(x)
        #print(f"x_encoded.shape: {x_encoded.shape}")
        #print(f"x_encoded[:10]: {x_encoded[:10]}")

        # Encode the time input t
        t_encoded = self.encode_time(t)
        #print(f"t_encoded.shape: {t_encoded.shape}")
        #print(f"t_encoded[:10]: {t_encoded[:10]}")

        # Concatenate x and the encoded t along the feature axis (second axis)
        xt = torch.cat([x_encoded, t_encoded], dim=1)
        #print(f"xt.shape: {xt.shape}")
        #print(f"xt[:10]: {xt[:10]}")

        # Perform pass through the network
        h = self.linear_1(xt)
        h = self.activation_fn(h)
        h = self.linear_2(h)
        h = self.activation_fn(h)
        h = self.linear_3(h)

        # Apply a softmax function to obtain class probabilities
        probs = self.softmax_fn(h)

        return probs
    
class PropertyModel(torch.nn.Module):
    def __init__(self, x_enc_dim, t_dim=1, y_dim=1, lat_dim=100):
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.x_enc_dim = x_enc_dim
        self.t_dim     = t_dim
        self.y_dim     = y_dim
        self.lat_dim   = lat_dim

        # Define the input and output dimension, based on the
        self.input_dim  = x_enc_dim + t_dim

        # Define an encoding for the x-values (i.e. categoricals)
        spatial_encoding  = encodings.OneHotEncoding1D(dim=self.x_enc_dim)
        self.encode_space = lambda x: spatial_encoding(x)

        # Define an encoding for the times (passed as scalars)
        #time_encoding = FourierEncoding(dim=self.t_dim)
        time_encoding = encodings.MultiLinearEncoding1D(dim=self.t_dim)
        self.encode_time = lambda t: time_encoding(t)
        #self.encode_time = lambda t: t.reshape(-1, 1)

        # Define an encoding for the property y
        y_encoding = encodings.OneHotEncoding1D(dim=self.y_dim)
        self.encode_y = lambda y: y_encoding(y)

        # Define the model parts
        self.linear_1 = torch.nn.Linear(self.input_dim, self.lat_dim)
        self.linear_2 = torch.nn.Linear(self.lat_dim, self.y_dim)

        # Define an activation function
        self.activation_fn = torch.nn.ReLU()

        # Define the softmax function that should be applied along the
        # second axis (i.e. the feature axis)
        self.softmax_fn = torch.nn.Softmax(dim=1)

    
    def forward(self, x, t):
        """
        Define forward pass of the model. 
        
        Args:
            x (torch.tensor): Torch tensor holding the x-values of the data.
            t (torch.tensor): Time as torch tensor.
        
        """
        # Encode the spatial features of the data (i.e. x)
        x_encoded = self.encode_space(x)

        # Encode the time input t
        t_encoded = self.encode_time(t)
        #print(f"t_encoded.shape: {t_encoded.shape}")
        #print(f"t_encoded[:10]: {t_encoded[:10]}")

        # Concatenate x and the encoded t along the feature axis (second axis)
        xt = torch.cat([x_encoded, t_encoded], dim=1)

        # Perform pass through the network
        h = self.linear_1(xt)
        h = self.activation_fn(h)
        h = self.linear_2(h)

        # Use a softmax
        y  = self.softmax_fn(h)

        return y

    def log_prob(self, x, t, y_data):
        """
        Return the log probability given the data. 
        
        Args:
            x (torch.tensor): Torch tensor holding the x-values of the data.
            t (torch.tensor): Time as torch tensor.
            y_data (torch.tensor): Torch tensor holding the y-values of the data.

        Return:
            (torch.tensor): log-probability for each point in the batch as
                1D torch tensor of shape (batch_size, ).
        
        """
        # Encode the property input y
        y_data_encoded = self.encode_y(y_data)

        # Predict y for the inputs
        y_pred = self.forward(x, t)

        # print(f"y_data.shape: {y_data.shape}")
        # print(f"y_data[:10] {y_data[:10]}")
        # print(f"y_data_encoded.shape: {y_data_encoded.shape}")
        # print(f"y_data_encoded[:10] {y_data_encoded[:10]}")
        # print(f"y_pred.shape: {y_pred.shape}")
        # print(f"y_pred[:10] {y_pred[:10]}")

        # Calculate the categorical log-probability for each point and each
        # category/feature and then sum over the feature (i.e. the second) axis
        log_prob = torch.sum(y_data_encoded*torch.log(y_pred+1e-10), dim=1)

        # print(f"log_prob.shape: {log_prob.shape}")
        # print(f"log_prob[:10] {log_prob[:10]}")

        return log_prob