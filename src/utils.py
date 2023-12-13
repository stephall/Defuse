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

def slice_tensor(tensor, inds, axis=1):
        """ 
        Slice along the input tensor along the specified axis using the passed indices.

        Args:
            tensor (torch.tensor): Either 2D torch tensor of shape (d_0, d_1) or
                3D torch tensor of shape (d_0, d_1, d_2).
            inds (torch.tensor): Indices that should be used to slice the tensor 
                along the specified axis as 1D torch tensor of shape (d_0,).
            axis (int): Zero-based axis to be sliced along as integer in [1, D-1] 
                where D is the dimension of the tensor (i.e. number of axes).
                (Default: 1)

        Return:
            (torch.tensor): Sliced tensor of shape (d_0, ) if the input tensor was 2D.
                or of shape (d_0, d_1) [for axis=2] or of shape (d_0, d_2) [for axis=1]
                if the input tensor was 3D.
        
        """
        # Differ cases depending on the dimension of the tensor
        if tensor.dim()==2:
            # Check that axis=1
            if axis!=1:
                err_msg = f"The input 'axis' must be either 1 if the input 'tensor' is 2D, got 'axis={axis}' instead."
                raise ValueError(err_msg)
            
            # Reshape the indices from (d_0, ) to (d_0, 1)
            slicing_inds = inds.reshape(-1, 1)
        elif tensor.dim()==3:
            # Differ cases for the axis
            if axis==1:
                # Construct a slicing index tensor of shape (d_0, 1, d_2) that 'repeats' the values in inds
                # along the third axis.
                slicing_inds = inds.reshape(-1, 1, 1).expand(-1, 1, tensor.shape[2])
            elif axis==2:
                # Construct a slicing index tensor of shape (d_0, d_1, 1) that 'repeats' the values in inds
                # along the second axis.
                slicing_inds = inds.reshape(-1, 1, 1).expand(-1, tensor.shape[1], 1)
            else:
                err_msg = f"The input 'axis' must be either 1 or 2 if the input 'tensor' is 3D, got 'axis={axis}' instead."
                raise ValueError(err_msg)
        else:
            err_msg = f"The input 'tensor' must be a 2D or 3D torch tensor, got dimension '{tensor.dim()}' instead."
            raise ValueError(err_msg)

        # Gather (i.e. slice) along the specified axis using the slicing indices.
        # This will result in a 2D tensor of shape (d_0, 1) if tensor is 2D and 3D
        # tensor of shape (d_0, 1, d_2) [for axis=1] and (d_0, d_1, 1) [for axis=2]
        # if tensor is 3D.
        # Thus, squeeze the result of the gather operation to obtain the sought for
        # tensor that has one axis/dimension less than the original tensor.
        sliced_tensor = tensor.gather(axis, slicing_inds).squeeze()
        
        return sliced_tensor

def expaddlog(a, b, eps=1e-15):
    """ 
    Perform multiplication of two tensors a and b using the exp-log trick:
    a*b = sign(a)*sign(b)*exp( log(|a|) + log(|b|) )

    Args:
        a (torch.tensor): First input tensor.
        b (torch.tensor): Second input tensor.
        eps (float): Tiny value used for numerical stability in the logarithm.
            (Default: 1e-15)

    Return:
        (torch.tensor): Result of a*b as torch tensor.

    Remark: The shapes of a and b must be such that they can be multiplied/added.
    
    """
    # Determine the signs of all entries of both a and b that should be tensors
    # of the same shape as a and b, respectively.
    sign_a = torch.sign(a)
    sign_b = torch.sign(b)

    # Determine the absolute values of all entries of both a and b that should 
    # be tensors of the same shape as the a and b, respectively.
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)

    # Return a*b = sign(a)*sign(b)*exp( log(|a|) + log(|b|) ) where a tiny
    # epsilon is used within the logarithms for numerical stability.
    return sign_a*sign_b*torch.exp( torch.log(abs_a+eps)+torch.log(abs_b+eps) )

def expsublog(a, b, eps=1e-15):
    """ 
    Perform division of two tensors a and b using the exp-log trick:
    a/b = sign(a)*sign(b)*exp( log(|a|) - log(|b|) )

    Args:
        a (torch.tensor): First input tensor.
        b (torch.tensor): Second input tensor.
        eps (float): Tiny value used for numerical stability in the logarithm.
            (Default: 1e-15)

    Return:
        (torch.tensor): Result of a/b as torch tensor.

    Remark: The shapes of a and b must be such that they can be divided/subtracted.
    
    """
    # Determine the signs of all entries of both a and b that should be tensors
    # of the same shape as a and b, respectively.
    sign_a = torch.sign(a)
    sign_b = torch.sign(b)

    # Determine the absolute values of all entries of both a and b that should 
    # be tensors of the same shape as the a and b, respectively.
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)

    # Return a*b = sign(a)*sign(b)*exp( log(|a|) + log(|b|) ) where a tiny
    # epsilon is used within the logarithms for numerical stability.
    return sign_a*sign_b*torch.exp( torch.log(abs_a+eps)-torch.log(abs_b+eps) )