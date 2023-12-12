# Import public modules
import torch

class BaseEncoding1D(object):
    """
    Define the base class for any type of '1D encoding'.
    Consider a 1D encoding as an encoding that is applied to 1D features of points in a batch
    transforming these features from a tensor of shape (batch_size,) [or (batch_size,1)] to
    (batch_size, encoding_dim).
    """
    def __init__(self, dim=1):
        # Assign inputs to class attributes
        self.dim = dim

    def _check_input(self, x):
        # Check that the input 'x' (i.e. the to be encoded features for points in a batch) is either 
        # a 1D tensor -- of shape (batch_size, ) -- or a 2D tensor of shape (batch_size, 1).
        if x.dim()==1:
            # If x is 1D, do nothing.
            pass
        elif x.dim()==2:
            # If x is 2D, check that it is of shape (batch_size, 1)
            if x.shape[1]!=1:
                err_msg = f"If the input 'x' is a 2D tensor it must be of shape '(batch_size, 1)', got shape '{x.shape}' instead. "
                raise ValueError(err_msg)
        else:
            err_msg = f"The input 'x' must be either a 1D tensor of shape '(batch_size, )' or a 2D tensor of shape '(batch_size, 1)', got dimension '{x.dim()}' and shape '{x.shape}' instead. "
            raise ValueError(err_msg)

class MultiLinearEncoding1D(BaseEncoding1D):
    def __init__(self, *args, **kwargs):
        # Initialize base class
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        """
        Encode the input 'x' holding 1D features for points in a batch.

        Args:
            x (torch.tensor): To be encoded features as tensor of shape (batch_size,) or (batch_size, 1).

        Return:
            (torch.tensor): Encoded features of shape (batch_size, self.dim) where self.dim is the
                dimensions of the encoding.
        
        """        
        # Check the input, i.e. it is either 1D torch tensor of shape (batch_size, ) or a 2D torch tensor 
        # of shape (batch_size, 1)
        self._check_input(x)

        # Ensure that the input is a 2D tensor of shape (batch_size, 1)
        x = x.reshape(-1, 1)

        # Determine the batch size from x, which is of shape (batch_size, 1)
        #batch_size = x.shape[0]

        # Construct all indices (of the basis)
        inds = torch.arange(1, self.dim+1).reshape(1, -1)

        # Encode x
        x_enc = inds*(2*x-1)/self.dim

        return x_enc

class OneHotEncoding1D(BaseEncoding1D):
    def __init__(self, *args, **kwargs):
        # Initialize base class
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        """
        Encode the input 'x' holding 1D features for points in a batch.

        Args:
            x (torch.tensor): To be encoded features as tensor of shape (batch_size,) or (batch_size, 1).

        Return:
            (torch.tensor): Encoded features of shape (batch_size, self.dim) where self.dim is the
                dimensions of the encoding.
        
        """        
        # Check the input, i.e. it is either 1D torch tensor of shape (batch_size, ) or a 2D torch tensor 
        # of shape (batch_size, 1)
        self._check_input(x)

        # Determine the batch size from x, which is of shape (batch_size,)
        batch_size = x.shape[0]

        # Ensure that the input is a 1D tensor of shape (batch_size,)
        x = x.squeeze()        

        # Initialize the encoding as zeros tensor of shape (batch_size, self.dim)
        x_enc = torch.zeros(batch_size, self.dim)

        # print(f"x_enc.shape: {x_enc.shape}")
        # print(f"x_enc[:10]: {x_enc[:10]}")

        # Set the value index corresponding to x in the second axis of 
        # x_enc to 1 for each of the points (along the first axis)
        axis_0_inds = torch.arange(batch_size)
        x_enc[axis_0_inds, x] = 1

        return x_enc
    
class RandomFourierEncoding1D(object):
    def __init__(self, dim=1):
        raise NotImplementedError("'Random Fourier (Feature) Encoding' has not been implemented.")
        
        # # Assign inputs to class attributes
        # self.dim = dim
