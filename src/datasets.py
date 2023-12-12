# Import public modules
import torch

class DictDataset(torch.utils.data.Dataset):
    """
    Define a custom dataset that returns a dictionary with dictionary-keys 
    'x' and 'y' if sliced where the dictionary-values will correspond to the 
    sliced x and y data values.
    """
    def __init__(self, x, y=None):
        """
        Args:
            x (torch.tensor): 2D torch tensor of shape 
                (#datapoints, #x-features).
            y (torch.tensor or None): Torch tensor of shape 
                (datapoints, #y-features) or 
                case y will be made a 1D torch tensor )
        
        """
        # Check that x is a 2D torch tensor
        if torch.is_tensor(x)==False:
            err_msg = f"The input 'x' must be a 2D torch tensor, got type '{type(x)}' instead."
            raise TypeError(err_msg)

        if x.dim()!=2:
            err_msg = f"The input 'x' must be a 2D torch tensor, got dimension '{x.dim()}' instead."
            raise ValueError(err_msg)

        # If y is not None, check that it is a torch tensor
        if y is not None:
            if torch.is_tensor(y)==False:
                err_msg = f"The input 'y' must be a torch tensor, got type '{type(y)}' instead."
                raise TypeError(err_msg)

            # if y.dim()!=2:
            #     err_msg = f"The input 'y' must be a 2D torch tensor, got dimension '{y.dim()}' instead."
            #     raise ValueError(err_msg)

        # Assign x and y to the corresponding class attributes
        self.x = x
        self.y = y

    def __len__(self):
        """ Return the number of datapoints. """
        # Remark: self.x should have shape (#datapoints, #x-features)
        return self.x.shape[0]

    def __getitem__(self, ix):
        """ Implement slicing. """
        # Cast ix to a list if it is a tensor
        if torch.is_tensor(ix):
            ix = ix.tolist()        

        # Return a dictionary of the data
        if self.y is None:
            return {'x': self.x[ix]}
        else:
            return {'x': self.x[ix], 'y': self.y[ix]}