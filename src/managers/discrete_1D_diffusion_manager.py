# Import public modules
import torch
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

# Import custom modules
from src import encodings
from src import utils
from . import base_diffusion_manager

# Reload custom modules
reload(encodings)
reload(utils)
reload(base_diffusion_manager)

class Discrete1DDiffusionManager(base_diffusion_manager.BaseDiffusionManager):
    """
    Define a manager to handle continuous time diffusion in discrete 1D spaces.
    """
    # Define a tiny number to be used for numerical stability (i.e. when dividing or taking logarithms)
    _eps = 1e-10

    def __init__(self, *args, prob_vec_1=None, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        ###################################################################################################################################################################################################################################
        ### Parse input 'prob_vec_1'
        ###################################################################################################################################################################################################################################
        if prob_vec_1 is None:
            err_msg = f"For discrete diffusion the input 'prob_vec_1' -- that specifies the fully noised distribution 'p_1(z_1)=Cat(z_1|p_vec_1)' -- must be passed during initialization of the diffusion manager object."
            raise ValueError(err_msg)
        
        # Check that prob_vec_1 is either a torch tensor, numpy array, list, or tuple
        if not (torch.is_tensor(prob_vec_1) or isinstance(prob_vec_1, (np.ndarray, list, tuple))):
            err_msg = f"The input probability vector 'prob_vec_1' that specifies the fully noised distribution 'p_1(z_1)=Cat(z_1|p_vec_1)' must be a torch tensor, numpy array, list, or tuple. Got type '{type(prob_vec_1)}' instead."
            raise TypeError(err_msg)
        
        # Cast 'prob_vec_1' to a torch tensor
        prob_vec_1 = torch.tensor(prob_vec_1, dtype=torch.float)

        # Check that prob_vec_1 is a 1D tensor
        if prob_vec_1.dim()!=1:
            err_msg = f"The input probability vector 'prob_vec_1' that specifies the fully noised distribution 'p_1(z_1)=Cat(z_1|p_vec_1)' must be a 1D tensor, got dimension '{prob_vec_1.dim()}' instead."
            raise ValueError(err_msg)
        
        # Normalize prob_vec_1
        prob_vec_1 = prob_vec_1/torch.sum(prob_vec_1)

        # Assign inputs to class attributes
        self.prob_vec_1 = prob_vec_1
        if self._debug:
            print(f"self.prob_vec_1.shape: {self.prob_vec_1.shape}")
            print(f"self.prob_vec_1: {self.prob_vec_1}")
        ###################################################################################################################################################################################################################################
        # The base rate matrix R_b = Identity-A_1 is specified over the matrix
        # A_1=1_vec*prob_vec_1^T where prob_vec_1 is the probability vector intended 
        # for t=1 and 1_vec is a vector full of 1s [i.e. 1_vec^T=(1, ..., 1)].
        # Remark: In case that prob_vec_1 contains a single '1' (with all other entries being '0'),
        #         this will lead to a rate matrix that has a row with all '0' entries for the column
        #         that corresponds to the entry with a '1' in prob_vec_1.
        #         Consequently, the jump probability out of this column/state will be 0 to any other
        #         state, which contradicts the demand to 'have a jump to another state' (and of course
        #         leads to an ill-defined, i.e. unnormalized, probability distribution). To counteract
        #         this issue, add a tiny value to all entries of prob_vec_1 and normalize the resulting
        #         vector again. This correction is only applied to 'prob_vec_1' and not 'self.prob_vec_1'
        #         as 'self.prob_vec_1' is only used to sample z_1 from p_1(z_1) where we can easily sample
        #         from a probability vector containing a single '1'.
        _prob_vec_1 = prob_vec_1 #+ 1e-6
        _prob_vec_1 = _prob_vec_1/torch.sum(_prob_vec_1)
        if self._debug:
            print(f"_prob_vec_1.shape: {_prob_vec_1.shape}")
            print(f"_prob_vec_1: {_prob_vec_1}")
        A_1 = torch.ones(_prob_vec_1.shape[0], 1)*_prob_vec_1.reshape(1, -1)
        Id  = torch.eye(_prob_vec_1.shape[0])

        # Make the tensors (i.e. matrices) A_1 and Id of shape (self.x_enc_dim, self.x_enc_dim)
        # tensors of shape (1, self.x_enc_dim, self.x_enc_dim) and assign them to their
        # corresponding class attributes. Also define R_b = A_1 - Id
        self.A_1 = torch.unsqueeze(A_1, dim=0)
        self.Id  = torch.unsqueeze(Id, dim=0)
        self.R_b = self.A_1-self.Id
        if self._debug:
            print(f"self.R_b: {self.R_b}")

        # Ensure that the sum over each of the rows of R_b is 0
        # Remark: This is a condition for the rate matrix R_t, and because
        #         we use R_t = gamma'(t)R_b, this condition translates to a 
        #         condition for R_b.
        if torch.any(1e-5<torch.abs(torch.sum(self.R_b.squeeze(), dim=1))):
            err_msg = f"The rows of the (base) rate matrix must sum to 0, which is not the case for the base rate matrix\nR_b={self.R_b.squeeze()}"
            raise ValueError(err_msg)
        

        # Get the encoding dimension of the spatial categorical 
        # from 'self.prob_vec_1' that is of shape (dim[x_enc_dim],)
        self.x_enc_dim = self.prob_vec_1.shape[0]

        # Define a one-hot encoding for the x-values (i.e. categoricals)
        one_hot_encoding = encodings.OneHotEncoding1D(dim=self.x_enc_dim)
        self.encode_x    = lambda x: one_hot_encoding(x)

        # Initialize the class attribute corresponding to the temperature used
        # when using guided-sampling for the property(-guide) distribution to 1
        self._guide_temp = 1

    def set_guide_temp(self, guide_temp):
        """ 
        Set the value of the class attribute corresponding to the temperature 
        used when using guided-sampling for the property(-guide) distribution.
        """
        # Check that guide_inv_temp is a zero or a positive number
        if not isinstance(guide_temp, (int, float)):
            err_msg = f"The input 'guide_temp' must be zero or a positive number (int or float), got type '{type(guide_temp)}' instead."
            raise TypeError(err_msg)
        
        if guide_temp<0:
            err_msg = f"The guide temperature must be zero or a positive number (int or float), got value '{guide_temp}' instead."
            raise ValueError(err_msg)

        # Set the class attribute
        self._guide_temp = guide_temp


    def Q(self, t):
        """
        Return the transition matrix Q_t that specifies the noising distribution of z_t:
        p_t(z_t) = Cat(z_t|prob_vec_t=enc(x)*Q_t).
        """
        # Determine kappa(t) that is of shape (batch_size, 1)
        kappa_t = torch.sqrt(self.kappa2(t))
        #print(f"kappa_t.shape: {kappa_t.shape}")

        # Unsqueeze kappa(t) to shape (batch_size, 1, 1)
        kappa_t = torch.unsqueeze(kappa_t, dim=2)
        #print(f"kappa_t.shape: {kappa_t.shape}")
        #print(f"self.A_1.shape: {self.A_1.shape}")
        #print(f"self.Id.shape: {self.Id.shape}")

        # Return Q_t = kappa(t)*Id+[1-kappa(t)]*A_1
        # Remark: self.Id and self.A_1 are both of shape (1, self.x_enc_dim, self.x_enc_dim)
        return kappa_t*self.Id + (1-kappa_t)*self.A_1
    
    def R(self, t):
        """
        Return the rate matrix R_t=R_b*gamma'(t).
        """
        # Determine gamma'(t) that is of shape (batch_size, 1)
        deriv_gamma_t = self.deriv_gamma(t)
        #print(f"deriv_gamma_t.shape: {deriv_gamma_t.shape}")

        # Unsqueeze gamma'(t) to shape (batch_size, 1, 1)
        deriv_gamma_t = torch.unsqueeze(deriv_gamma_t, dim=2)
        #print(f"deriv_gamma_t.shape: {deriv_gamma_t.shape}")
        #print(f"self.R_b.shape: {self.R_b.shape}")

        # Return R_t = R_b*gamma'(t)
        # Remark: self.R_b is of shape (1, self.x_enc_dim, self.x_enc_dim)
        return deriv_gamma_t*self.R_b

    def diffusion_batch_loss(self, batch_data):
        """
        Return the batch loss for diffusion. 
        
        Args:
            batch_data (torch.utils.data.Dataset): Torch dataset object
                holding the data of the current batch that can be passed
                to the denoising model.

        Return:
            (torch.tensor): Scalar loss of the batch for the passed batch data.
        
        """
        # Access the batch x-data
        batch_x = batch_data['x']

        # Determine the batch size from batch_x that has shape
        # (batch_size, #x-features)
        batch_size = batch_x.shape[0]

        # Set the number of x-features if it is not defined
        if self.x_num_features is None:
            self.x_num_features = batch_x.shape[1]

        # Check that 'self.x_num_features' corresponds to batch_x.shape
        if self.x_num_features!=batch_x.shape[1]:
            err_msg = "The number of features of x has changed during training."
            raise ValueError(err_msg)

        # Sample a time t for each point in the batch
        batch_t = self._sample_times_over_batch(batch_size)
        if self._debug:
            print(f"batch_t.shape: {batch_t.shape}")
            print(f"batch_t: {batch_t[:10]}")
            print()

        # Sample latent state of each datapoint in the batch for their
        # corresponding time t (in forward diffusion process)
        batch_z_t = self.forward_sample_z_t(batch_x, batch_t)
        if self._debug:
            print(f"batch_z_t.shape: {batch_z_t.shape}")
            print(f"batch_z_t[:10]:  {batch_z_t[:10]}")
            print()

        # Determine the (batched) forward transition rate vector R_t(z_t, :)
        # that will be a 2D torch tensor of shape (batch_size, self.x_enc_dim).
        batch_R_t_z_t_vec = self._get_R_t_z_start_t_vec(batch_z_t, batch_t)

        # Determine the (batched) forward normalizing constant Z_t(z_t)
        # that will be a 1D torch tensor of shape (batch_size,)
        batch_Z_t_z_t = self._get_Z_t_z_t(batch_z_t, batch_R_t_z_t_vec)

        # Determine the (batched) probability vector to jump from z_t to any other state
        # that will be a 2D torch tensor of shape (batch_size, self.x_enc_dim).
        prob_vec_jump_z_t = self._get_jump_prob_vec(batch_z_t, batch_R_t_z_t_vec)

        # Sample z_tilde_t from a categorical distribution with the probability vector p^{jump}_t[z_t]
        # that corresponds to p^{jump}_t(z_tilde_t|z_t).
        q_z_tilde_t_z_t = torch.distributions.categorical.Categorical(prob_vec_jump_z_t) 
        batch_z_tilde_t = q_z_tilde_t_z_t.sample()
        if self._debug:
            print(f"batch_z_tilde_t.shape: {batch_z_tilde_t.shape}")
            print(f"batch_z_tilde_t[:10]:  {batch_z_tilde_t[:10]}")
            print()

        # Determine \hat{R}(z^{start}_t, :) where the starting state z^{start}_t is z_tilde_t here
        # as 2D tensor of shape (batch_size, self.x_enc_dim) corresponding to the inverse (i.e. back
        # ward in time) transition rates from start state z^{start}_t to any other end state z^{end}_t
        batch_hat_R_hat_t_z_tilde_t_vec = self._predict_hat_R_t_z_start_t_vec(batch_z_tilde_t, batch_t, train_or_eval='train', batch_y=None)

        # Slice 'batch_R_hat_t_z_tilde_t_vec', i.e. \hat{R}_t(z_tilde_t, z), which is a 2D torch tensor of shape 
        # (batch_size, self.x_enc_dim), along the last axis (axis=1 because axis is zero based) by fixing these
        # indices to 'batch_z_t' (i.e. z=z_t) resulting in a 1D torch tensor of shape (batch_size,)
        batch_hat_R_t_z_tilde_t_z_t = utils.slice_tensor(batch_hat_R_hat_t_z_tilde_t_vec, batch_z_t, axis=1)

        # Determine the (batched) backward normalizing constant \hat{Z}_t(z_tilde_t)
        # that will be a 1D torch tensor of shape (batch_size,)
        batch_hat_Z_hat_t_z_tilde_t = self._get_Z_t_z_t(batch_z_tilde_t, batch_hat_R_hat_t_z_tilde_t_vec)
        if self._debug:
            print(f"batch_hat_Z_hat_t_z_tilde_t: {batch_hat_Z_hat_t_z_tilde_t.shape}")
            print(f"batch_hat_Z_hat_t_z_tilde_t[:10]: {batch_hat_Z_hat_t_z_tilde_t[:10]}")
            print()

        # Calculate the loss function
        # \hat{Z}_t(\tilde{z}_t) - Z_t(z_t)*log[\hat{R}^theta_t(\tilde{z}_t, z_t)]
        # for each point in the batch (i.e. a the result is a loss vector over the batch)
        # Remark: Add a tiny value to the logarithm input for numerical stability.
        vectorial_batch_loss_t = batch_hat_Z_hat_t_z_tilde_t - batch_Z_t_z_t*torch.log(batch_hat_R_t_z_tilde_t_z_t+self._eps)
        if self._debug:
            print(f"vectorial_batch_loss_t.shape: {vectorial_batch_loss_t.shape}")
            print(f"vectorial_batch_loss_t[:10]:  {vectorial_batch_loss_t[:10]}")
            print()

        # Get the time-dependent weight for the loss function
        batch_time_weight_t = self.time_weight(batch_t)

        # Sum over the entries of this loss vector
        batch_loss_t = torch.sum(batch_time_weight_t*vectorial_batch_loss_t)
        if self._debug:
            print(f"batch_loss_t.shape: {batch_loss_t.shape}")
            print(f"batch_loss_t:  {batch_loss_t}")
            print()

        #raise ValueError("AM HERE")
    
        return batch_loss_t
    
    def _get_Z_t_z_t(self, batch_z_t, batch_R_t_z_t_vec):
        """
        Return the total rate to transition out from (i.e. jump) a state 'z_t'
        to any other state z at time t, i.e. Z_t(z_t) = \sum_{z!=z_t} R_t(z_t, z).

        Remark: This method works both for forward or backward time rates!

        Args:
            batch_z_t (torch.tensor): The batched state from which to jump 
                from as 2D torch tensor of shape (batch_size, #x-features).
            batch_R_t_z_t_vec (torch.tensor): The transition rates R_t(z_t, z) 
                at time t from z_t to any state z represented as vector 
                R_t(z_t, :) for each point in the batch in the form of a torch 
                tensor of shape (batch_size, self.x_enc_dim).

        Return:
            (torch.tensor): The jump rates out of state 'z_t' for each point in 
                the batch as 1D tensor of shape (batch_size,).

        """
        # Slice 'batch_R_t_z_t' that is now of shape (batch_size, self.x_enc_dim) once more along the last axis 
        # (axis=1 because axis is zero based), which is former axis=2 before the last slicing operation, by also 
        # fixing these indices to the sampled 'batch_z_t' resulting in a 1D torch tensor of shape (batch_size,)
        batch_R_t_z_t_z_t = utils.slice_tensor(batch_R_t_z_t_vec, batch_z_t, axis=1)
        # if self._debug:
        #     print(f"batch_R_t_z_t_z_t.shape: {batch_R_t_z_t_z_t.shape}")
        #     print(f"batch_R_t_z_t_z_t[:10]:  {batch_R_t_z_t_z_t[:10]}")
        #     print()

        # Determine the normalizing constant Z_t(z_t) = sum_{c!=z_{t}}R_t(z_t, c) (where c is some integer representing 
        # a categorical of the state space) by using that the rows of R_t must sum to 0, i.e. sum_{c}R_t(z_t, c), s.t. 
        # Z_t(z_t) = -R_t(z_t, z_t).
        batch_Z_t_z_t = -batch_R_t_z_t_z_t
        if self._debug:
            print(f"batch_Z_t_z_t.shape: {batch_Z_t_z_t.shape}")
            print(f"batch_Z_t_z_t[:10]:  {batch_Z_t_z_t[:10]}")
            print()

        return batch_Z_t_z_t
    
    def _get_jump_prob_vec(self, batch_z_start_t, batch_R_t_z_start_t_vec):
        """
        Return the jump probabilities from a state z^{start}_t to any other
        state z^{end}_t!=z^{start}_t.
        
        Remark: This method works both for forward or backward time rates!

        Args:
            batch_z_start_t (torch.tensor): The batched start state to jump 
                from as 2D torch tensor of shape (batch_size, #x-features).
            batch_R_t_z_start_t_vec (torch.tensor): The transition rates
                R_t(z^{start}_t, z) at time t from z^start_t to any state z
                represented as vector R_t(z^{start}_t, :) for each point in the 
                batch in the form of a torch tensor of shape (batch_size, self.x_enc_dim).
        
        Return:
            (torch.tensor): The jump probabilities from state z^{start}_t at time t
                as 2D tensor of shape (batch_size, self.x_enc_dim)

        """
        # Calculate the jump-probabilities 
        # p^{jump}_t[z^{start}_t](c) = [1-delta_{z^{start}_t, c}] R_t(z^{start}_t, c)/Z_t(z^{start}_t)
        # (where c is some integer representing a categorical of the state space) with the normalizing
        # constant 
        # Z_t(z_t) = \sum_{c'!=z^{start}_t} R_t(z^{start}_t, c') 
        #          = \sum_{c'!=z^{start}_t} [1-delta_{z^{start}_t, c'}]R_t(z^{start}_t, c')
        # Thus, we can also write
        # w^{jump}_t[z^{start}_t](c) = [1-delta_{z^{start}_t, c}] R_t(z^{start}_t, c)
        # p^{jump}_t[z^{start}_t](c) = w^{jump}_t[z^{start}_t](c)/( \sum_{c'} w^{jump}_t[z^{start}_t](c') )
        # Remarks: (1) delta_{z_t, :} corresponds (in tensorial form) to encode(z_t) because
        #              encoded(z_t) will be of shape (batch_size, self.x_enc_dim) with a 1 in 
        #              each row at the position corresponding to z_t.
        #          (2) Add tiny value to 'batch_R_t_z_start_t_vec' when calculating the weights for numerical 
        #              stability (i.e. avoiding division by zero).
        batch_delta_z_start_t       = self.encode_x(batch_z_start_t)
        weight_vec_jump_t_z_start_t = (1-batch_delta_z_start_t)*(batch_R_t_z_start_t_vec+self._eps)
        Z_t_z_start_t               = torch.sum(weight_vec_jump_t_z_start_t, dim=1)
        prob_vec_jump_t_z_start_t   = weight_vec_jump_t_z_start_t/( Z_t_z_start_t.reshape(-1, 1) )
        if self._debug:
            print(f"prob_vec_jump_t_z_start_t.shape: {prob_vec_jump_t_z_start_t.shape}")
            print(f"prob_vec_jump_t_z_start_t[:10]: {prob_vec_jump_t_z_start_t[:10]}")
            print(f"sum[:10]: {torch.sum(prob_vec_jump_t_z_start_t, dim=1)}")
            print()

        return prob_vec_jump_t_z_start_t
    
    def _get_R_t_z_start_t_vec(self, batch_z_start_t, batch_t):
        """
        Return the forward rate vector R_t(z^{start}_t, z^{end}_t) that contains 
        the rates to transition from z^{start}_t to state z^{end}_t at time t, where
        the entries of \hat{R}_t(z^{start}_t, z^{end}_t) correspond to each of the z^{end}_t,
        for one fixed starting state z^{start}_t.

        Remark: z^{start}_t and z^{end}_t are both states at the same time that are separated
                by a 'forward in time transition' from z^{start}_t to z^{end}_t.
                The equivalent 'backward in time transition' would be from z^{end}_t to z^{start}_t.
        
        Args:
            batch_z_start_t (torch.tensor): The batched start state to jump from as 2D torch 
                tensor of shape (batch_size, #x-features).
            batch_t (torch.tensor): The batched time at which the jump occurs (for each point 
                in the batch) as 2D torch tensor of shape (batch_size, 1).
        
        Return:
            (torch.tensor): The batched forward transition rate vector R_t(z_t, :)
                as 2D torch tensor of shape (batch_size, self.x_enc_dim).

        """
        # Determine the R_t matrix for the batch times
        batch_R_t = self.R(batch_t)
        if self._debug:
            print(f"batch_R_t.shape: {batch_R_t.shape}")
            print(f"batch_R_t[:10]:  {batch_R_t[:10]}")
            print()

        # Slice 'batch_R_t' by fixing the second axis (axis=1 because axis is zero-based) indices to the
        # values of 'batch_z_start_t' that will lead to a 2D torch tensor of shape (batch_size, self.x_enc_dim).
        # Remarks: (a) 'batch_z_start_t' is a 1D torch tensor of shape (batch_size,).
        #          (b) 'batch_R_t' is a 3D torch tensor of shape (batch_size, self.x_enc_dim, self.x_enc_dim).
        #          (c) I.e. obtain the 2D torch tensor [R_t]_{:, batch_z_start_t, :} of shape (batch_size, self.x_enc_dim).
        batch_R_t_z_start_t_vec = utils.slice_tensor(batch_R_t, batch_z_start_t, axis=1)
        if self._debug:
            print(f"batch_R_t_z_start_t_vec.shape: {batch_R_t_z_start_t_vec.shape}")
            print(f"batch_R_t_z_start_t_vec[:10]:  {batch_R_t_z_start_t_vec[:10]}")
            print()

        return batch_R_t_z_start_t_vec
    
    def _predict_hat_R_t_z_start_t_vec(self, batch_z_start_t, batch_t, train_or_eval, batch_y=None):
        """
        Predict the inverse/backward rate vector \hat{R}_t(z^{start}_t, z^{end}_t) that contains 
        the rates to transition from z^{start}_t to state z^{end}_t at time t, where
        the entries of \hat{R}_t(z^{start}_t, z^{end}_t) correspond to each of the z^{end}_t,
        for one fixed starting state z^{start}_t.

        Remarks: 
        z^{start}_t and z^{end}_t are both states at the same time that are separated        
        by a 'backward in time transition' from z^{start}_t to z^{end}_t.
        The equivalent 'forward in time transition' would be from z^{end}_t to z^{start}_t.
        
        Args:
            batch_z_start_t (torch.tensor): The batched start state to jump from as 2D torch tensor 
                of shape (batch_size, #x-features).
            batch_t (torch.tensor): The batched time at which the jump occurs (for each point in the 
                batch) as 2D torch tensor of shape (batch_size, 1).
            train_or_eval (str): The 'train' or 'eval' mode to be used for the denoising model
                when predicting the inverse/backward rate vector.
            batch_y (None or torch.tensor): If not None, the property y as 2D torch tensor 
                of shape (1, #y-features).
                (Default: None)
        
        Return:
            (torch.tensor): The batched inverse/backward transition rate vector \hat{R}_t(z^{start}_t, :)
                as 2D torch tensor of shape (batch_size, self.x_enc_dim).

        """
        # Determine the Q_t and R_t matrices for the batch times
        batch_Q_t = self.Q(batch_t)
        batch_R_t = self.R(batch_t)
        if self._debug:
            print(f"batch_Q_t.shape: {batch_Q_t.shape}")
            print(f"batch_Q_t[:10]:  {batch_Q_t[:10]}")
            print(f"batch_R_t.shape: {batch_R_t.shape}")
            print(f"batch_R_t[:10]:  {batch_R_t[:10]}")
            print()
        
        # Slice Q_t and R_t by fixing the last (i.e. third) axis (axis=2 because axis is zero-based) 
        # indices to the entries of 'batch_z_start_t' that will lead to 2D torch tensors of 
        # shape (batch_size, self.x_enc_dim).
        # Remarks: (a) 'batch_z_start_t' is a 1D torch tensor of shape (batch_size,)
        #          (b) 'batch_Q_t' and 'batch_R_t' are a 3D torch tensors of shape (batch_size, self.x_enc_dim, self.x_enc_dim)
        #          (c) I.e. obtain 2D torch tensor [Q_t]_{:, :, batch_z_start_t} and [R_t]_{:, :, batch_z_start_t}.
        batch_Q_t_vec_z_start_t = utils.slice_tensor(batch_Q_t, batch_z_start_t, axis=2)
        batch_R_t_vec_z_start_t = utils.slice_tensor(batch_R_t, batch_z_start_t, axis=2)
        
        if self._debug:
            print(f"batch_Q_t_vec_z_start_t.shape: {batch_Q_t_vec_z_start_t.shape}")
            print(f"batch_Q_t_vec_z_start_t[:10]:  {batch_Q_t_vec_z_start_t[:10]}")
            print(f"batch_R_t_vec_z_start_t.shape: {batch_R_t_vec_z_start_t.shape}")
            print(f"batch_R_t_vec_z_start_t[:10]:  {batch_R_t_vec_z_start_t[:10]}")
            print()

        
        # Construct the model input for the current batch
        if batch_y is not None:
            batch_model_input = {'x': self.encode_x(batch_z_start_t), 'y': batch_y}
        else:
            batch_model_input = {'x': self.encode_x(batch_z_start_t)}

        # Set the denoising model into 'train mode' or 'eval mode'
        if train_or_eval=='train':
            self.model_dict['denoising'].train()
        elif train_or_eval=='eval':
            self.model_dict['denoising'].eval()
        else:
            err_msg = f"The input 'train_or_eval' must be either 'train' or 'eval'."
            raise ValueError(err_msg)

        # Use the prediction model to predict p^{theta}_{0|t}(z'_t|z_t) 
        # for fixed z^{start}_t and free z'_t, resulting in a tensor
        # of shape (batch_size, self.x_enc_dim)
        batch_p_theta_vec_z_start_t = self.model_dict['denoising'](batch_model_input, batch_t)

        # Calculate the fraction fraction_pq = p^{\theta}_{0|t}(z'_t|z^{start}_t)/q_{t|0}(z^{start}_t|z'_t)
        # Remark: q_{t|0}(z^{start}_t|z'_t) corresponds to [Q_t]_{z'_t, z^{start}_t} (in unbatched form).
        # Thus, perform element-wise division of batch_p_theta_z_start_t and batch_Q_t_z_start_t
        # (corresponding to [Q_t]_{:, z'_t, z^{start}_t}, i.e. q_{t|0}(z^{start}_t|z'_t) in batched form)
        # for fixed z^{start}_t and free z'_t, i.e. [Q_t]_{z'_t, z^{start}_t} has 
        # shape (batch_size, self.x_enc_dim), so the same shape as p^theta_{0|t}.
        # Remarks: (1) batch_Q_t_z_start_t corresponds to [Q_t]_{:, :, batch_z_start_t}.
        #          (2) Use a/b = exp(log(a)-log(b)) trick for numerical stability.
        batch_fraction_pq_t = utils.expsublog(batch_p_theta_vec_z_start_t, batch_Q_t_vec_z_start_t, eps=self._eps)
        if self._debug:
            print(f"batch_fraction_pq_t.shape: {batch_fraction_pq_t.shape}")
            print(f"batch_fraction_pq_t[:10]:  {batch_fraction_pq_t[:10]}")
            print()

        # Determine the sum 
        # sum_qpq
        # \sum_{z'_t} q_{t|0}(z^{end}_t|z'_t)*p^{\theta}_{0|t}(z'_t|z^{start}_t)/q_{t|0}(z^{start}_t|z'_t)
        # = 
        # \sum_{z'_t} q_{t|0}(z^{end}_t|z'_t)*fraction_pq
        # where fraction_pq = p^{\theta}_{0|t}(z'_t|z^{start}_t)/q_{t|0}(z^{start}_t|z'_t) 
        # has been determined above.
        # Element-wise multiply 'batch_fraction_pq_t' of shape (batch_size, self.x_enc_dim)
        # with batch_Q_t of shape (batch_size, self.x_enc_dim, self.x_enc_dim), which corresponds
        # to q_{t|0}(:|:), while summing over the second axis of both tensors.
        # TODO: MAKE THIS NUMERICALLY MORE ROBUST!!!! (Multiplication and then summation)
        batch_sum_qpq_t = torch.einsum('bij,bi->bj', batch_Q_t, batch_fraction_pq_t)
        if self._debug:
            print(f"batch_sum_qpq_t.shape: {batch_sum_qpq_t.shape}")
            print(f"batch_sum_qpq_t[:10]:  {batch_sum_qpq_t[:10]}")
            print()

        # To obtain the batched 'inverse/backward rate' \hat{R}^{theta}_t(z^{start}_t, z^{end}_t) with fixed start 
        # state z^{start}_t and arbitrary end state z^{end}_t -- i.e. a 3D tensor of shape (batch_size, self.x_enc_dim, 
        # self.x_enc_dim) sliced (along the last/third axis) to a 2D tensor of shape (batch_size, self.x_enc_dim) -- 
        # we element-wise multiply the 'forward rate' R_t(z^{end}_t, z^{start}_t), which describes the 'forward in time
        # transition' from z^{end}_t to z^{start}_t, corresponding to the vectorial batch_R_t_z_start_t [=R_t(:,z^{start}_t)]
        # with batch_sum_qpq_t corresponding to 
        # \sum_{z'_t} q_{t|0}(z^{end}_t|z'_t)*p^{\theta}_{0|t}(z'_t|z^{start}_t)/q_{t|0}(z^{start}_t|z'_t)
        # Remarks: (1) Use the 'a*b = exp(log(a)+log(b))' trick for numerical stability.
        #          (2) 'batch_hat_R_t_z_start_t_vec' will be a tensor of shape (batch_size, self.x_enc_dim)
        batch_hat_R_t_z_start_t_vec = utils.expaddlog(batch_R_t_vec_z_start_t, batch_sum_qpq_t, eps=self._eps)
        if self._debug:
            print(f"batch_z_start_t.shape: {batch_z_start_t.shape}")
            print(f"batch_z_start_t[:10]:  {batch_z_start_t[:10]}")
            print(f"[raw] batch_hat_R_t_z_start_t_vec.shape: {batch_hat_R_t_z_start_t_vec.shape}")
            print(f"[raw] batch_hat_R_t_z_start_t_vec[:10]:  {batch_hat_R_t_z_start_t_vec[:10]}")
            print()

        # Calculating the batched 'inverse/backward rate' \hat{R}^{theta}_t(z^{start}_t, z^{end}_t) above is only valid
        # for all elements with z^{start}_t!=z^{end}_t. The elements with z^{start}_t==z^{end}_t correspond to
        # \hat{R}^{theta}_t(z^{start}_t, z^{start}_t) = \sum_{z'_t!=z^{start}_t} \hat{R}^{theta}_t(z^{start}_t, z'_t)
        # which can also be written as
        # \hat{R}^{theta}_t(z^{start}_t, z^{start}_t) = \hat{R}^{theta}_t(z^{start}_t, z^{start}_t)-\sum_{z'_t} \hat{R}^{theta}_t(z^{start}_t, z'_t)
        # To only apply the correction to the entries of \hat{R}^{theta}_t(z^{start}_t, z) with z=z^{start}_t, we
        # can use a binary mask for the correction, i.e. 
        # \hat{R}^{theta}_t(z^{start}_t, z) = \hat{R}^{theta}_t(z^{start}_t, z)-\delta_{z^{start}_t, z} \sum_{z'_t} \hat{R}^{theta}_t(z^{start}_t, z'_t)
        # where \delta_{z^{start}_t, z} is a 2D torch tensor that has a 1 at location z^{start}_t for each point in the batch
        # and corresponds to the one-hot encoding of z^{start}_t.
        # Step 1: Obtain the binary mask
        batch_delta_z_start_t_vec = self.encode_x(batch_z_start_t)
        # Step 2: Apply correction
        batch_hat_R_t_z_start_t_vec = batch_hat_R_t_z_start_t_vec - batch_delta_z_start_t_vec*torch.sum(batch_hat_R_t_z_start_t_vec, dim=1).reshape(-1, 1)
        if self._debug:
            print(f"batch_delta_z_start_t_vec.shape: {batch_delta_z_start_t_vec.shape}")
            print(f"batch_delta_z_start_t_vec[:10]: {batch_delta_z_start_t_vec[:10]}")
            print(f"[corrected] batch_hat_R_t_z_start_t_vec.shape: {batch_hat_R_t_z_start_t_vec.shape}")
            print(f"[corrected] batch_hat_R_t_z_start_t_vec[:10]:  {batch_hat_R_t_z_start_t_vec[:10]}")
            print(f"Sum[:10]: {torch.sum(batch_hat_R_t_z_start_t_vec, dim=1)[:10]}")
            print()

        return batch_hat_R_t_z_start_t_vec


    def forward_sample_z_t(self, x, t):
        """
        Sample z_t in forward diffusion (i.e. noised state at time t).
        
        Args:
            x (torch.tensor): Original (i.e. unnoised) x-data as 2D tensor
                of shape (batch_size, #x-features).
            t (torch.tensor, float or int): Time to sample z_t at.

        Return:
            (torch.tensor): Sampled z_t for time 't' and original x-data 'x'
                as 2D tensor of shape (batch_size, #x-features).

        """
        # Get batch size
        batch_size = x.shape[0]
        #print(f"x.shape: {x.shape}")
        #print(f"x[:10]:  {x[:10]}")

        # One hot encode the data
        x_enc = self.encode_x(x)
        # if self._debug:
        #     print(f"x_enc.shape: {x_enc.shape}")
        #     print(f"x_enc[:10]:  {x_enc[:10]}")
        #     print()

        # Parse the time t
        t = self._parse_time(t, batch_size)
        #print(f"t.shape: {t.shape}")

        # Determine Q_t for these times
        Q_t = self.Q(t)
        # if self._debug:
        #     print(f"Q_t.shape: {Q_t.shape}")
        #     print(f"Q_t: {Q_t}")
        #     print()

        # Determine the probability vector of the categorical distribution at time t
        # Remarks: (1) x_enc has shape (batch_size, self.x_enc_dim)
        #          (2) Q_t has shape (batch_size, self.x_enc_dim, self.x_enc_dim)
        #          (3) Use matrix multiplication of features in x_enc (second axis) 
        #              and matrices in Q_t (second and third axes):
        #              sum_{i=1}^{self.x_enc_dim} [x_enc]_(b,i)[Q_t]_(b,i,j)
        #              where b is the index of a point in the batch.
        prob_vec_t = torch.einsum('bi,bij->bj', x_enc, Q_t)
        if self._debug:
            print(f"prob_vec_t.shape: {prob_vec_t.shape}")
            print(f"prob_vec_t[:10]:  {prob_vec_t[:10]}")
            print()
        
        # Sample z_t from a categorical distribution with the probability vector prob_vec_t
        q_z_t = torch.distributions.categorical.Categorical(prob_vec_t) 
        z_t   = q_z_t.sample()
        if self._debug:
            print(f"z_t.shape: {z_t.shape}")
            print(f"z_t[:10] : {z_t[:10]}")

        return z_t

    def generate(self, batch_size=100, max_num_time_steps=100, num_integration_points=100, random_seed=24, y=None, guide_temp=1):
        """
        Generate 'novel' points by sampling from p(z_1) and then using ancestral 
        sampling (backwards in time) to obtain 'novel' \hat{z}_0=\hat{x}.

        Args:
            batch_size (int): Number of 'novel' points '\hat{x}' to generate.
                (Default: 100)
            max_num_time_steps (int): Maximal number of ancestral sampling (time) 
                steps to use for backward propagation through time.
                (Default: 100)
            num_integration_points (int): Number of integration points used to
                approximate the integral int_{t_last}^{t_next}\hat{Z}_s(z_t_last)ds
                when sampling the next time t_next.
                (Default: 100)
            random_seed (int): Random seed to be used for all the sampling in
                the generation process.
                (Default: 24)
            y (int or None): Conditional class to guide to.
        
        Return:
            (torch.tensor): 'Novel' points as 2D torch tensor \hat{x} of shape
                (batch_size, #x-features) generated in backward diffusion.
        
        """
        # Set the property(-guide) distribution temperature
        self.set_guide_temp(guide_temp)

        # Parse y if it is not None
        if y is not None:
            y = y*torch.ones(batch_size, dtype=torch.int).reshape(-1, 1)

        # Set a random seed
        self.set_random_seed(random_seed)

        # We don't need to calculate any gradients for discrete generation
        with torch.no_grad():
            # Individually generate every point of the batch
            x_generated_list = list()
            t_end_list       = list()
            for batch_point in range(batch_size):
                # Generate a single point that will be a 2D tensor of shape (1, self.x_enc_dim).
                # Remark: The method '_generate_single_point' will return the last time step
                x_generated, t_end = self._generate_single_point(max_num_time_steps=max_num_time_steps, 
                                                                num_integration_points=num_integration_points, 
                                                                y=y[batch_point])

                # If the last time step is 0, append the generated x
                if t_end==0:
                    x_generated_list.append(x_generated)

        # Cast the generated points to a 2D tensor of shape (#batch_size-#non_converged, self.x_enc_dim)
        # where #non_converged is the number of points for which the final time was not 0
        batch_x_generated = torch.cat(x_generated_list, dim=0)

        # Inform the user in case that the number of generated points is less than the requested batch size
        if batch_x_generated.shape[0]!=batch_size:
            print(f"Could only generate {batch_x_generated.shape[0]/batch_size*100: 0.1f}% of the required points.")
            plt.figure()
            plt.hist(t_end_list, bins=30, density=True, color='b', alpha=0.5)
            plt.xlabel(r'$t_{end}$')
            plt.ylabel(r'$p(t_{end})$')
            plt.show()

        # Return the final z_t
        return batch_x_generated
    
    def _generate_single_point(self, max_num_time_steps=100, num_integration_points=100, y=None):
        """
        Generate one single 'novel' point by sampling from p(z_1) and then using ancestral 
        sampling (backwards in time) to obtain the 'novel' \hat{z}_0=\hat{x}.

        Remarks:
        (1) Use backward sampling scheme proposed in 
            'A Continuous Time Framework for Discrete Denoising Models'
            (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al.,
            where this sampling here corresponds to the first step where
            the next time 't_next' (or equialently a holding time 
            t_last-t_next) is sampled.

        (2) Perform next=time sampling using the time-dependent Gillespie algorithm
            'A modified Next Reaction Method for simulating chemical systems with time dependent 
            propensities and delays' (https://arxiv.org/abs/0708.0370) by D. F. Anderson.

        Args:
            max_num_time_steps (int): Maximal number of ancestral sampling (time) 
                steps to use for backward propagation through time.
                (Default: 100)
            num_integration_points (int): Number of integration points used to
                approximate the integral int_{t_last}^{t_next}\hat{Z}_s(z_t_last)ds
                when sampling the next time t_next.
                (Default: 100)
            y (int or None): Conditional class to guide to.
        
        Return:
            (torch.tensor): 'Novel' point as 2D torch tensor \hat{x} of shape
                (1, #x-features) generated in backward diffusion.
        
        """
        # Initialize the current (i.e. last) time 't_last' as 2D torch tensor of shape (1, 1) containing the value 1
        # Remark: We always use times as 2D tensors of shape (batch_size, 1) but for the purpose here the batch_size is 1.
        t_last = torch.tensor(1).reshape(1, 1)

        # Sample z_(t=1)=z_1 and set it as the initial current (i.e. last) state 'z_t_last'
        z_t_last = self._sample_z_1(batch_size=1).reshape(1, -1)
        if self._debug:
            print(f"z_1.shape: {z_t_last.shape}")
            print(f"z_1[:10]:  {z_t_last[:10]}")

        # Loop over the time steps
        for _ in range(max_num_time_steps):
            # Sample the next time, i.e. by holding the state z_t_last for the
            # holding time equals to t_last-t_next
            # Remark: We sample backward in time, so t_last>t_next
            t_next = self._backward_sample_t_next(t_last, z_t_last, num_integration_points=num_integration_points)
            if self._debug:
                print(f"t_last.shape: {t_last.shape}")
                print(f"t_last:       {t_last}")
                print(f"t_next.shape: {t_next.shape}")
                print(f"t_next:       {t_next}")

            # If this new time 't_next' is 0, the state z_t_last is kept until time t=0 and we can 
            # therefore stop the generation process.
            if t_next.item()==0:
                # Return z_t_last (that is the final state at t=0 here) and t_next
                return z_t_last, t_next.item()

            # Sample the next state 'z_t_next' obtained in a jump from the current 
            # (i.e. last) state 'z_t_last' at time 't_next'.
            z_t_next = self._backward_sample_z_t_next(z_t_last, t_next, y=y)

            # Make the next state and time the last state and time for the next iteration
            # Remark: Explicit is better than implicit
            t_last   = t_next
            z_t_last = z_t_next

        # Return the final z_t_next and t_next
        return z_t_next, t_next.item()
    
    def _backward_sample_t_next(self, t_last, z_t_last, num_integration_points=100):
        """
        Sample the next time (backward in time) 't_next' based on the current (i.e. last) 
        time 't_last' and the current (i.e. last) state at this time 'z_t_last'.
        The state z_t_last stays the same for the holding time given by the difference
        of t_last and t_next.

        Remarks:
        (1) Use backward sampling scheme proposed in 
            'A Continuous Time Framework for Discrete Denoising Models'
            (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al.,
            where this sampling here corresponds to the first step where
            the next time 't_next' (or equialently a holding time 
            t_last-t_next) is sampled.

        (2) Perform the first part of the time-dependent Gillespie algorithm
            in this method using 'A modified Next Reaction Method for simulating 
            chemical systems with time dependent propensities and delays' 
            (https://arxiv.org/abs/0708.0370) by D. F. Anderson.
            => Sample u from Uniform(0, 1)
            => Find t_next that fulfills -log(u)=int_{t_last}^{t_next}\hat{Z}_s(z_t_last)ds.

        (3) t_next<t_last because we are sampling backwards in time.
        
        Args:
            t_last (torch.tensor): The current (i.e. last) time as 2D torch tensor
                of shape (1, 1).
            z_t_last (torch.tensor): The current (i.e. last) state at time 't_last'
                that is holded until the next time 't_next' (sampled here) as
                2D torch tensor of shape (1, #x-features). 
            num_integration_points (int): Number of integration points used to
                approximate the integral int_{t_last}^{t_next}\hat{Z}_s(z_t_last)ds
                when sampling the next time t_next.
                (Default: 100)

        Return:
            (torch.tensor): Sampled next time as 2D torch tensor of shape (1, 1).
        
        """
        
        # Expand z_t_last, which is a 0D tensor to the number of integration steps
        # so that each integration step has the same z_t
        expanded_z_t_last = z_t_last.expand(num_integration_points, -1)
        if self._debug:
            print(f"expanded_z_t_last.shape: {expanded_z_t_last.shape}")
            print(f"expanded_z_t_last[:10]: {expanded_z_t_last[:10]}")

        # Determine the time integration points {t_j}_{j=0}^{N-1} where t_0=t_last and t_{N-1}=0 
        # Remark: t_last is a 2D tensor of shape (1, 1), use its value [].item()] as float for torch.linspace.
        t_intpnts = torch.linspace(t_last.item(), 0, num_integration_points).reshape(-1, 1)

        # Also determine the absolute values of their time-differences, i.e. t_diff_j = |t_{j+1}-t_{j}|
        # Remark: As we are going backward in time 'torch.diff(t_intpnts.squeeze())' only
        #         contains negative values
        t_diff = torch.abs( torch.diff(t_intpnts.squeeze()) )
        if self._debug:
            print(f"t_intpnts.shape: {t_intpnts.shape}")
            print(f"t_intpnts[:10]:  {t_intpnts[:10]}")
            print(f"t_diff.shape: {t_diff.shape}")
            print(f"t_diff[:10]:  {t_diff[:10]}")


        # Determine the backward transition rate vector \hat{R}_{t_intpnts}(z_t_last, :) for the 
        # same z_t_last value at each of the time integration points (i.e. the expanded one)
        hat_R_t_intpnts_z_t_last_vec = self._predict_hat_R_t_z_start_t_vec(expanded_z_t_last, t_intpnts, train_or_eval='eval')
        #if self._debug:
        #print(f"batch_hat_R_t_z_t_vec.shape: {batch_hat_R_t_z_t_vec.shape}")
        #print(f"batch_hat_R_t_z_t_vec[:10]:  {batch_hat_R_t_z_t_vec[:10]}")

        # Determine \hat{Z}_t(z_t_last)=\sum_{z!=z_t_last} \hat{R}_t(z_t_last, z) that corresponds 
        # to the sum over the backward rates of all transitions that jump out of the 
        # current z_t_last (i.e. the total rate to transition/jump out of the current z_t_last)
        # at each of the time integration points
        hat_Z_t_intpnts_z_t_last = self._get_Z_t_z_t(expanded_z_t_last, hat_R_t_intpnts_z_t_last_vec)
        if self._debug:
            print(f"hat_Z_t_intpnts_z_t_last.shape: {hat_Z_t_intpnts_z_t_last.shape}")
            print(f"hat_Z_t_intpnts_z_t_last[:10]:  {hat_Z_t_intpnts_z_t_last[:10]}")
            print(f"hat_Z_t_intpnts_z_t_last[-10:]: {hat_Z_t_intpnts_z_t_last[-10:]}")

        # Numerically integrate (using the trapezoidal rule) \hat{Z}_s(z_t_last) in every 
        # time-interval bounded by the time integration points, i.e. determine the integrals
        # int_{t_j}^{t_{j+1}}\hat{Z}_s(z_t_last)ds
        integ_subint_t_intpnts = (hat_Z_t_intpnts_z_t_last[1:]+hat_Z_t_intpnts_z_t_last[:-1])/2*t_diff
        if self._debug:
            print(f"integ_subint_t_intpnts.shape: {integ_subint_t_intpnts.shape}")
            print(f"integ_subint_t_intpnts[:10]:  {integ_subint_t_intpnts[:10]}")

        # Cummulating the intergrals of these time-intervals determine the numerical 
        # approximation of the integral from t_last up to any of the time integration 
        # points, i.e. int_{t_last}^{t_j}\hat{Z}_s(z_t_last)ds for any j>0:
        integ_t_intpnts = torch.cumsum(integ_subint_t_intpnts, dim=0)

        # At t_0=t_last, the integral is 0 because int_{a}^{a}f(s)ds=0 for any function f(s) and 
        # any integration boundary point a. Thus, add 0 as the first integral entry for the integral
        # up to t_0 (=t_last)
        integ_t_intpnts = torch.cat([torch.tensor(0).reshape(-1, 1), integ_t_intpnts.reshape(-1, 1)]).squeeze()
        if self._debug:
            print(f"t_intpnts[:10]: {t_intpnts[:10]}")
            print(f"[raw]integ_t_intpnts.shape: {integ_t_intpnts.shape}")
            print(f"[raw]integ_t_intpnts[:10]:  {integ_t_intpnts[:10]}")
            print(f"[updated]integ_t_intpnts.shape: {integ_t_intpnts.shape}")
            print(f"[updated]integ_t_intpnts[:10]:  {integ_t_intpnts[:10]}")
            print(f"[updated]integ_t_intpnts[-10:]:  {integ_t_intpnts[-10:]}")

        # Draw i.i.d. uniform random variable from Uniform(0, 1)
        u_sample = torch.rand(1)

        # Create a mask for the time intervals [t_i, t_{i+1}] (i.e. the index i) for which 
        # int_{t}^{t_i}\hat{Z}_{s}(z_t)ds<=-log(u)<int_{t}^{t_{i+1}}\hat{Z}_{s}(z_t)ds
        mask = torch.logical_and(integ_t_intpnts[:-1]<=-torch.log(u_sample), -torch.log(u_sample)<integ_t_intpnts[1:])
        
        # Differ the cases where one interval exists that fulfills this condition or not
        if torch.all(mask==False):
            # If all entries of the mask are False, no such interval exists and thus 
            # int_{t}^{0}\hat{Z}_{s}(z_t)ds<-log(u)
            # so that the sampled holding time must exceed time 0 (i.e. t-t_hold<0).
            # Thus, set t_next to 0
            t_next = torch.tensor(0)
        else:
            # Otherwise, one interval [t_i, t_{i+1}] existed where the condition above
            # is fulfilled and thus the mask has a True entry.
            # Get the index 'i' of this interval
            i = torch.where(mask)[0]

            # Get the time associated with this i
            t_i = t_intpnts[i]

            # Calculate D = -log(u)-int_{t}^{t_i}\hat{Z}_{s}(z_t)ds=-[ log(u) + int_{t}^{t_i}\hat{Z}_{s}(z_t)ds ]
            D = (-torch.log(u_sample)-integ_t_intpnts[i]).squeeze()
            if self._debug:
                print(f"D.shape: {D.shape}")
                print(f"D:  {D}")

            # Extract \hat{Z}_{t_{i}}(z_t) and \hat{Z}_{t_{i+1}}(z_t)
            hat_Z_t_i_z_t_last   = hat_Z_t_intpnts_z_t_last[i]
            hat_Z_t_ip1_z_t_last = hat_Z_t_intpnts_z_t_last[i+1]
            if self._debug:
                print(f"hat_Z_t_i_z_t_last.shape: {hat_Z_t_i_z_t_last.shape}")
                print(f"hat_Z_t_i_z_t_last[:10]:  {hat_Z_t_i_z_t_last}")
                print(f"hat_Z_t_i+1_z_t_last.shape: {hat_Z_t_ip1_z_t_last.shape}")
                print(f"hat_Z_t_i+1_z_t_last[:10]:  {hat_Z_t_ip1_z_t_last}")

            # Calculate delta_hat_Z_t_i_z_t = [\hat{Z}_{t_{i+1}}(z_t)-\hat{Z}_{t_{i}}(z_t)]/Delta_t_i
            # Remark: As we will divide by this value, use add a tiny value for numerical stability.
            delta_hat_Z_t_i_z_t = (hat_Z_t_ip1_z_t_last-hat_Z_t_i_z_t_last)/t_diff[i]+self._eps

            # Calculate d_i = t_i-t_next and then determine t_next from it
            d_i    = ( -hat_Z_t_i_z_t_last + torch.sqrt(hat_Z_t_i_z_t_last**2+2*D*delta_hat_Z_t_i_z_t) )/delta_hat_Z_t_i_z_t
            t_next = (t_i - d_i).squeeze()
            if self._debug:
                print(f"d_i: {d_i}")
                print(f"Delta_t_i: {t_diff[i]}")
                print(f"t_i: {t_i}")

        # Plot the quantities if debugging
        if self._debug:
            print(f"t_next: {t_next}")
            print()

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            ax = axs[0, 0]
            ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), hat_Z_t_intpnts_z_t_last.cpu().detach().numpy(), 'bo-')
            ax.set_ylabel(r'$Z_{t}(z_{t})$')

            ax = axs[1, 0]
            ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), (1/hat_Z_t_intpnts_z_t_last).cpu().detach().numpy(), 'bo-')
            ax.set_ylabel(r'$\frac{1}{Z_{t}(z_{t})}$')

            ax = axs[0, 1]
            ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), integ_t_intpnts.cpu().detach().numpy(), 'bo-')
            ax.set_ylabel(r'$\int_{t_{last}}^{t}Z_{s}(z_{s})ds$')
            ax.set_xlabel(r'$t$')

            ax = axs[1, 1]
            ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), (1/integ_t_intpnts).cpu().detach().numpy(), 'bo-')
            ax.set_ylabel(r'$\frac{t_{last}}{\int_{1}^{t}Z_{s}(z_{s})ds}$')
            ax.set_xlabel(r'$t$')
            
            plt.show()

        return t_next.reshape(1, 1)

    def _backward_sample_z_t_next(self, z_t_last, t_next, y=None):
        """
        Sample next state 'z_t_next' backward in time from last (i.e. current) state
        'z_t_last' at time 't_next' where the two states are connected via a jump
        (i.e. z_t_last!=z_t_next).
        
        Remark:
        Use backward sampling scheme proposed in 
        'A Continuous Time Framework for Discrete Denoising Models'
        (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al.,
        where this sampling here corresponds to the second part where 
        'z_t_next' is sampled as a state jumped to from 'z_t_last' (!=z_t_next)
        at time 't_next' (after holding for the time t_last-t_next).
        => The input 't_last' is not used within this method but kept for 
           consistency with the corresponding method in the continuous space case.

        Args:
            z_t_last (torch.tensor): The last (i.e. current) z_t as 2D torch 
                tensor of shape (1, #x-features).
            t_next (torch.tensor): The next times as 2D torch tensor of shape (1, 1).
            y (None or torch.tensor): If not None, the property y as 2D torch tensor 
                of shape (1, #y-features).
                (Default: None)

        Return:
            (torch.tensor): Sampled 'z_t_next' as torch tensor of the
                same shape as the last (i.e. current) 'z_t_last'.

        """
        # Determine the backward transition rate vector \hat{R}_{t_{next}}(z_t, :)
        hat_R_t_next_z_t_vec = self._predict_hat_R_t_z_start_t_vec(z_t_last, t_next, train_or_eval='eval')
        if self._debug:
            print(f"hat_R_t_next_z_t_vec.shape: {hat_R_t_next_z_t_vec.shape}")
            print(f"hat_R_t_next_z_t_vec[:10]:  {hat_R_t_next_z_t_vec[:10]}")

        # Determine the probability vector to jump from z_t_last to a new state at time t_next
        prob_vec_jump_t_next_z_t_last = self._get_jump_prob_vec(z_t_last, hat_R_t_next_z_t_vec)
        if self._debug:
            print(f"prob_vec_jump_t_next_z_t_last.shape: {prob_vec_jump_t_next_z_t_last.shape}")
            print(f"prob_vec_jump_t_next_z_t_last: {prob_vec_jump_t_next_z_t_last}")

        ###################################################################################################
        ### Property guidance - start
        ###################################################################################################
        # In case that the property model exists (i.e. is not None), determine 
        # the gradient of it w.r.t. to z_t_last
        if self.model_dict['property'] is not None:   
            if y is None:
                err_msg = f"Can only use property guidance in case that 'y' is passed to 'generate' method."
                raise ValueError(err_msg)
            
            # Create a torch tensor with all the transition states that can be accessed from z_t_last
            # as 2D torch tensor of shape (#transitions, #x-features), which included z_t_last itself.
            # Encode this transition index.
            z_t_from_z_t_last         = torch.arange(self.x_enc_dim).reshape(-1, 1)
            z_t_from_z_t_last_encoded = self.encode_x(z_t_from_z_t_last)
            if self._debug:
                print(f"z_t_from_z_t_last.shape: {z_t_from_z_t_last.shape}")
                print(f"z_t_from_z_t_last[:10]:  {z_t_from_z_t_last[:10]}")
                print(f"z_t_from_z_t_last_encoded.shape: {z_t_from_z_t_last_encoded.shape}")
                print(f"z_t_from_z_t_last_encoded[:10]:  {z_t_from_z_t_last_encoded[:10]}")
                print(f"y.shape: {y.shape}") 
                print(f"y: {y}")

            # Expand the next time to 
            expanded_t_next = t_next.expand(z_t_from_z_t_last_encoded.shape[0], -1)
            if self._debug:
                print(f"expanded_t_next.shape: {expanded_t_next.shape}")
                print(f"expanded_t_next[:10]:  {expanded_t_next[:10]}")

            # Expand the y values
            expanded_y = y.expand(z_t_from_z_t_last_encoded.shape[0], -1)
            if self._debug:
                print(f"expanded_y.shape: {expanded_y.shape}")
                print(f"expanded_y[:10]:  {expanded_y[:10]}")

            # Determine the log-probability (i.e. log-likelihood) of the property 
            # model for each of the jump states. The result 'log_prob' will be a 
            # 1D torch tensor of shape (batch_size,)
            log_prob = self.model_dict['property'].log_prob(z_t_from_z_t_last_encoded, expanded_t_next, expanded_y)
            if self._debug:
                print(f"log_prob.shape: {log_prob.shape}")
                print(f"log_prob[:10]:  {log_prob[:10]}")
                print(f"prob.shape:     {torch.exp(log_prob).shape}")
                print(f"prob[:10]:      {torch.exp(log_prob)[:10]}")
                print(f"[raw]prob_vec_jump_t_next_z_t_last.shape: {prob_vec_jump_t_next_z_t_last.shape}")
                print(f"[raw]prob_vec_jump_t_next_z_t_last[:10]:  {prob_vec_jump_t_next_z_t_last[:10]}")

            # Update the transition probabilities
            # (i.e. actually the jump probabilities because the probability to transition to the same state is 0)
            # by multiplying them with the property(-guide) distribution (i.e. the likelihood)
            # p^{jump-updated}(z|z_t_last) = [p^{jump}(z|z_t_last)*p^{property}(y|z)]/[ Sum_{z} p^{jump}(z|z_t_last)*p^{property}(y|z) ]
            # Differ cases depending on the guide temperature
            if self._guide_temp==0:
                # If the temperature guide distribution is 0, the jump probability to the state with the 
                # maximum of the property(-guide) distribution (leaving out the state corresponding to 
                # z_t_last) is 1 and all the other probabilities are zero.
                # First, determine the probabilities
                prob = torch.exp(log_prob)

                # First, set the log-probabilities corresponding to z_t_last to 0.
                delta_z_t_next_z_t_last = self.encode_x(z_t_last)
                prob                    = (1-delta_z_t_next_z_t_last)*prob

                # Second, update 'prob_vec_jump_t_next_z_t_last' to be a zeros tensor that has
                # only a 1 at for the state-entry (along second axis) that corresponds to the 
                # maximum of prob.
                ix = torch.argmax(prob).reshape(1, -1)
                prob_vec_jump_t_next_z_t_last        = torch.zeros_like(prob_vec_jump_t_next_z_t_last)
                prob_vec_jump_t_next_z_t_last[:, ix] = 1

            else:
                # Case where the temperature of the guide distribution is finite
                # First, calculate the element-wise log-product using the log(a*b)=log(a)+log(b) trick
                # Remark: We do not use 'utils.expaddlog' here because log_prob is already a logarithm and because
                #         here a and b are both non-negative (so we do not have to take care of their sign).
                #         Use tiny value in logarithm for numerical stability.
                #prob_vec_jump_t_next_z_t_last = torch.exp( torch.log(prob_vec_jump_t_next_z_t_last+self._eps) + log_prob )
                #prob_vec_jump_t_next_z_t_last = utils.expaddlog(prob_vec_jump_t_next_z_t_last, torch.exp(log_prob/self._guide_temp))
                log_prob_vec_jump_t_next_z_t_last = torch.log(prob_vec_jump_t_next_z_t_last+self._eps) + log_prob/self._guide_temp

                # Second, shift the log-probabilities so that their maximum is 0 (i.e. the corresponding non-log value is 1)
                # and compute the exponential.
                # Remark: Perform 'maximum shift trick' for numerical stability as exp(f(x))/sum_{x'}exp(f(x')) is invariant under any shifts.
                #         When calculating the maximum, calculate it over the state (i.e. second) axis as 'prob_vec_jump_t_next_z_t_last'
                #         and thus also its logarithm will have shape (1, self.x_enc_dim).
                log_prob_vec_jump_t_next_z_t_last = log_prob_vec_jump_t_next_z_t_last - torch.max(log_prob_vec_jump_t_next_z_t_last, dim=1).values
                prob_vec_jump_t_next_z_t_last     = torch.exp(log_prob_vec_jump_t_next_z_t_last)

                # Third, set the probabilities corresponding to z_t_last to 0
                # Remark: Not using the a*b=exp(log(a)+log(b)) trick would ensure this trivially
                #         as the non-updated jump probability will be zero for z_t_last.
                #         Although 'utils.expaddlog' might also return a zero value for z_t_last,
                #         it is good idea to 'hard enforce' this here explicitly.
                delta_z_t_next_z_t_last       = self.encode_x(z_t_last)
                prob_vec_jump_t_next_z_t_last = (1-delta_z_t_next_z_t_last)*prob_vec_jump_t_next_z_t_last
            
                # Fourth, normalize the probabilities
                prob_vec_jump_t_next_z_t_last = prob_vec_jump_t_next_z_t_last/torch.sum(prob_vec_jump_t_next_z_t_last)
            
            # Display results for the user
            if self._debug:
                print(f"[guided]prob_vec_jump_t_next_z_t_last.shape: {prob_vec_jump_t_next_z_t_last.shape}")
                print(f"[guided]prob_vec_jump_t_next_z_t_last[:10]:  {prob_vec_jump_t_next_z_t_last[:10]}")

            #raise ValueError("AM HERE")

        ###################################################################################################
        ### Property guidance -end
        ###################################################################################################

        # Sample z_t_next from a categorical distribution with the probability vector p^{jump}_{t_next}[z_t_last]
        # that corresponds to p^{jump}_{t_next}(z^{jump}_t_{next}|z_t_last).
        # I.e. z^{jump}_t_next is obtained by a jump from z_t_last at time t_next.
        q_z_jump_t_next_z_t = torch.distributions.categorical.Categorical(prob_vec_jump_t_next_z_t_last) 
        z_jumpt_t_next      = q_z_jump_t_next_z_t.sample()
        if self._debug:
            print(f"z_t_last.shape: {z_t_last.shape}")
            print(f"z_t_last[:10]:  {z_t_last[:10]}")
            print(f"z_jumpt_t_next.shape: {z_jumpt_t_next.shape}")
            print(f"z_jump_t_next_t[:10]:  {z_jumpt_t_next[:10]}")
            print()

        return z_jumpt_t_next
    
    def _sample_z_1(self, batch_size):
        """
        I.i.d. drawn z_1 for each point in a batch.

        Args:
            batch_size (int): Batch size.

        Return:
            (torch.tensor): 1D torch tensor corresponding to a z_1 sample that
                will have shape (batch_size,).
        
        """
        # Use the same p_1 probability vector for each point in the batch.
        # Thus, repeat self.prob_vec_1, which is of shape (self.x_enc_dim, ),
        # along a new first axis (corresponding to the batch axis) to obtain
        # a torch tensor of shape (batch_size, self.x_enc_dim) that has
        # self.prob_vec_1 in each of its rows
        batch_prob_vec_1 = self.prob_vec_1.repeat(batch_size, 1)

        # Define a p_1(z_1) as categorical distribution with the p_1 probability vector
        # Remark: The sampled z_1 is a 1D torch tensor of shape (batch_size,) containing
        #         integers (i.e. categoricals)
        batch_p_1_distr = torch.distributions.categorical.Categorical(batch_prob_vec_1) 
        batch_z_1       = batch_p_1_distr.sample()

        return batch_z_1
    
    def _get_property_model_log_prob(self, batch_z_t, batch_t, batch_y):
        """
        Return the log probability of the property log model.
        
        Args:
            batch_z_t (torch.tensor): (Batched) state z_t as 2D torch tensor
                of shape (batch_size, 1).
            batch_t (torch.tensor): (Batched) time as 2D torch tensor
                of shape (batch_size, 1).
            batch_y (torch.tensor): (Batched) property values as 2D torch tensor
                of shape (batch_size, #y-features).

        Return:
            (torch.tensor): Log probability of the property model for the inputs
                as 1D torch tensor of shape (batch_size,)
        
        """
        # Set the property model into 'train mode'
        self.model_dict['property'].train()

        # Encode z_t
        batch_z_t_encoded = self.encode_x(batch_z_t)
        
        # Determine the log-probability (i.e. log-likelihood) of the property 
        # model for the batch data for each point in the batch, i.e. log_prob 
        # is a 1D torch tensor of shape (batch_size,)
        return self.model_dict['property'].log_prob(batch_z_t_encoded, batch_t, batch_y)
    
###################################################################################################
### DISCRETE DIFFUSION METHODS SIDETRACK
###################################################################################################

# def plot_t_next_distribution(self, batch_size=100, num_integration_points=100, random_seed=24, y=None):
#         """
        
#         """
#         # Parse y if it is not None
#         if y is not None:
#             y = y*torch.ones(batch_size, dtype=torch.int).reshape(-1, 1)

#         # Set a random seed
#         self.set_random_seed(random_seed)

#         # Set the last (i.e. current) time to 1
#         t_last = torch.tensor(1)-1

#         # Sample a single z_(t=1)=z_1, expand it set it as the initial z_t
#         #z_1_sample = self._sample_z_1(batch_size=1)
#         z_1_sample = torch.tensor(1).reshape(-1, 1)
#         z_t        = z_1_sample.expand(num_integration_points, -1)


#         # Determine the time integration points {t_j}_{j=0}^{N-1} where t_0=t_last and t_{N-1}=0 
#         # and the absolute values of their time-differences, i.e. t_diff_j = |t_{j+1}-t_{j}|
#         t_intpnts = torch.linspace(t_last, 0, num_integration_points).reshape(-1, 1)
#         t_diff    = torch.abs(torch.diff(t_intpnts.squeeze()))

#         if self._debug:
#             print(f"t_intpnts.shape: {t_intpnts.shape}")
#             print(f"t_intpnts[:10]:  {t_intpnts[:10]}")
#             print(f"t_diff.shape: {t_diff.shape}")
#             print(f"t_diff[:10]:  {t_diff[:10]}")

#         #if self._debug:
#         print(f"z_1.shape: {z_t.shape}")
#         print(f"z_1[:10]:  {z_t[:10]}")

#         # Determine the backward transition rate vector \hat{R}_{t_intpnts}(z_t, :) for the same z_t
#         # at each of the time integration points
#         hat_R_t_intpnts_z_t_vec = self._predict_hat_R_t_z_start_t_vec(z_t, t_intpnts, train_or_eval='eval')
#         #if self._debug:
#         #print(f"batch_hat_R_t_z_t_vec.shape: {batch_hat_R_t_z_t_vec.shape}")
#         #print(f"batch_hat_R_t_z_t_vec[:10]:  {batch_hat_R_t_z_t_vec[:10]}")

#         # Determine \hat{Z}_t(z_t)=\sum_{z!=z_t} \hat{R}_t(z_t, z) that corresponds 
#         # to the sum over the backward rates of all transitions that jump out of the 
#         # current z_t (i.e. the total rate to transition/jump out of the current z_t)
#         # at each of the time integration points
#         hat_Z_t_intpnts_z_t = self._get_Z_t_z_t(z_t, hat_R_t_intpnts_z_t_vec)
#         #if self._debug:
#         print(f"hat_Z_t_z_t.shape: {hat_Z_t_intpnts_z_t.shape}")
#         print(f"hat_Z_t_z_t[:10]:  {hat_Z_t_intpnts_z_t[:10]}")
#         print(f"hat_Z_t_z_t[-10:]: {hat_Z_t_intpnts_z_t[-10:]}")

#         # Interpolate between these points
#         num_interpol_steps = int(1e4) #num_integration_points
#         t_interpol = np.linspace(t_last.item(), 0, num_interpol_steps)
#         print(t_interpol.shape)
#         print(t_intpnts.cpu().detach().numpy().squeeze().shape)
#         print(hat_Z_t_intpnts_z_t.cpu().detach().numpy().squeeze().shape)
#         print(t_interpol)
#         print(t_intpnts.cpu().detach().numpy().squeeze())
#         print(hat_Z_t_intpnts_z_t.cpu().detach().numpy().squeeze())
#         hat_Z_t_interpol_z_t = np.interp(t_interpol[::-1], t_intpnts.cpu().detach().numpy().squeeze()[::-1], hat_Z_t_intpnts_z_t.cpu().detach().numpy().squeeze()[::-1])[::-1]
#         diff_t_interpol = np.abs(np.diff(t_interpol))
#         integ_subint_hat_Z_t_interpol_z_t = (hat_Z_t_interpol_z_t[1:]+hat_Z_t_interpol_z_t[:-1])/2*diff_t_interpol

#         print(f"hat_Z_t_interpol_z_t.shape: {hat_Z_t_interpol_z_t.shape}")
#         print(f"hat_Z_t_interpol_z_t[:10]:  {hat_Z_t_interpol_z_t[:10]}")
#         print(f"hat_Z_t_interpol_z_t[-10:]: {hat_Z_t_interpol_z_t[-10:]}")

#         # Numerically integrate (using the trapezoidal rule) every time-interval bounded
#         # by the time integration points, i.e. every intergral between t_j and t_{j+1}
#         integ_subint_t_intpnts = (hat_Z_t_intpnts_z_t[1:]+hat_Z_t_intpnts_z_t[:-1])/2*t_diff
#         print(f"integ_subint_t_intpnts.shape: {integ_subint_t_intpnts.shape}")
#         print(f"integ_subint_t_intpnts[:10]:  {integ_subint_t_intpnts[:10]}")

#         # Cummulating the intergrals of these time-intervals determine the numerical 
#         # approximation of the integral from t_last up to any of the time integration 
#         # points, i.e. int_{t_last}^{t_j} for any j>0:
#         integ_t_intpnts = torch.cumsum(integ_subint_t_intpnts, dim=0)
#         print(f"[raw]integ_t_intpnts.shape: {integ_t_intpnts.shape}")
#         print(f"[raw]integ_t_intpnts[:10]:  {integ_t_intpnts[:10]}")

#         integ_hat_Z_t_interpol_z_t = np.cumsum(integ_subint_hat_Z_t_interpol_z_t)
#         integ_hat_Z_t_interpol_z_t = np.hstack([0, integ_hat_Z_t_interpol_z_t])
#         print(f"t_interpol[:20]: {t_interpol[:20]}")
#         print(f"integ_hat_Z_t_interpol_z_t.shape: {integ_hat_Z_t_interpol_z_t.shape}")
#         print(f"integ_hat_Z_t_interpol_z_t[:20] {integ_hat_Z_t_interpol_z_t[:20]}")

#         # At t_0, the integral is 0, so add this integral as the first integral entry
#         integ_t_intpnts = torch.cat([torch.tensor(0).reshape(-1, 1), integ_t_intpnts.reshape(-1, 1)]).squeeze()
#         print(f"t_intpnts[:10]: {t_intpnts[:10]}")
#         print(f"[updated]integ_t_intpnts.shape: {integ_t_intpnts.shape}")
#         print(f"[updated]integ_t_intpnts[:10]:  {integ_t_intpnts[:10]}")
#         print(f"[updated]integ_t_intpnts[-10:]:  {integ_t_intpnts[-10:]}")
#         print(f"[updated]max(integ_t_intpnts):  {torch.max(integ_t_intpnts)}")
#         print(f"zipped: {list(zip(integ_t_intpnts[:-1], integ_t_intpnts[1:]))}")

#         # Draw i.i.d. uniform random variables
#         #u_sample = torch.rand(1)
#         u_samples = torch.rand(int(1e4))


#         ix = np.argmin(np.abs(integ_hat_Z_t_interpol_z_t.reshape(-1, 1)+torch.log(u_samples).cpu().detach().numpy().reshape(1, -1)), axis=0)
#         print(ix.shape)
#         t_next_array = t_interpol[ix]
#         print(f"t_next_array.shape: {t_next_array.shape}")
#         print(f"t_next_array[:10]: {t_next_array[:10]}")
#         print("HERE")


#         t_next_list = list()
#         for u_sample in u_samples:
#             #u_sample = 1/torch.exp(torch.tensor(21))
#             # print(f"u_sample.shape: {u_sample.shape}")
#             # print(f"u_sample: {u_sample}")

#             # Create a mask for the time intervals [t_i, t_{i+1}] (i.e. the index i) for which 
#             # int_{t}^{t_i}\hat{Z}_{s}(z_t)ds<=-log(u)<int_{t}^{t_{i+1}}\hat{Z}_{s}(z_t)ds
#             mask = torch.logical_and(integ_t_intpnts[:-1]<=-torch.log(u_sample), -torch.log(u_sample)<integ_t_intpnts[1:])
            
#             # Differ the cases where one interval exists that fulfills this condition or not
#             if torch.all(mask==False):
#                 # If all entries of the mask are False, no such interval exists and thus 
#                 # int_{t}^{0}\hat{Z}_{s}(z_t)ds<-log(u)
#                 # so that the sampled holding time must exceed time 0 (i.e. t-t_hold<0).
#                 # Thus, set t_next to 0
#                 t_next = torch.tensor(0)
#             else:
#                 # Otherwise, one interval [t_i, t_{i+1}] existed where the condition above
#                 # is fulfilled and thus the mask has a True entry.
#                 # Get the index 'i' of this interval
#                 i = torch.where(mask)[0]

#                 # Get the time associated with this i
#                 t_i = t_intpnts[i]

#                 # Calculate D = -log(u)-int_{t}^{t_i}\hat{Z}_{s}(z_t)ds=-[ log(u) + int_{t}^{t_i}\hat{Z}_{s}(z_t)ds ]
#                 D = (-torch.log(u_sample)-integ_t_intpnts[i]).squeeze()
#                 # print(f"D.shape: {D.shape}")
#                 # print(f"D:  {D}")

#                 # Extract \hat{Z}_{t_{i}}(z_t) and \hat{Z}_{t_{i+1}}(z_t)
#                 hat_Z_t_i_z_t   = hat_Z_t_intpnts_z_t[i]
#                 hat_Z_t_ip1_z_t = hat_Z_t_intpnts_z_t[i+1]
#                 # print(f"hat_Z_t_i_z_t.shape: {hat_Z_t_i_z_t.shape}")
#                 # print(f"hat_Z_t_i_z_t[:10]:  {hat_Z_t_i_z_t}")
#                 # print(f"hat_Z_t_i+1_z_t.shape: {hat_Z_t_ip1_z_t.shape}")
#                 # print(f"hat_Z_t_i+1_z_t[:10]:  {hat_Z_t_ip1_z_t}")

#                 # Calculate delta_hat_Z_t_i_z_t = [\hat{Z}_{t_{i+1}}(z_t)-\hat{Z}_{t_{i}}(z_t)]/Delta_t_i
#                 # Remark: As we will divide by this value, use 'eps' to ensure numerical stability
#                 delta_hat_Z_t_i_z_t = (hat_Z_t_ip1_z_t-hat_Z_t_i_z_t)/t_diff[i]+self._eps

#                 # Calculate \delta_t_i = t_i-t_next and then determine t_next from it
#                 delta_t_i = ( -hat_Z_t_i_z_t + torch.sqrt(hat_Z_t_i_z_t**2+2*D*delta_hat_Z_t_i_z_t) )/delta_hat_Z_t_i_z_t
#                 t_next    = (t_i - delta_t_i).squeeze()

#                 # print(delta_t_i)
#                 # print(t_diff[i])
#                 # print(t_i)
#             #print(t_next)

#             t_next_list.append(t_next.reshape(-1, 1))

#         t_next_tensor = torch.cat(t_next_list)

#         plt.figure()
#         plt.hist(t_next_tensor.detach().cpu().numpy(), bins=100, density=True, color='b', alpha=0.5)
#         plt.hist(t_next_array, bins=100, density=True, color='r', alpha=0.5)
#         plt.xlim([0, 1])
#         plt.xlabel('t_next')
#         plt.show()


#         #plt.figure(figsize=(7, 7))
#         fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#         ax = axs[0, 0]
#         ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), hat_Z_t_intpnts_z_t.cpu().detach().numpy(), 'bo', ms=0.5)
#         ax.plot(t_interpol, hat_Z_t_interpol_z_t, 'b-', lw=0.2)
#         ax.set_ylabel(r'$Z_{t}(z_{t})$')

#         ax = axs[1, 0]
#         ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), (1/hat_Z_t_intpnts_z_t).cpu().detach().numpy(), 'b-')
#         ax.set_ylabel(r'$\frac{1}{Z_{t}(z_{t})}$')

#         ax = axs[0, 1]
#         ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), integ_t_intpnts.cpu().detach().numpy(), 'bo', ms=0.5)
#         ax.plot(t_interpol, integ_hat_Z_t_interpol_z_t, 'b-', lw=0.2)
#         ax.set_ylabel(r'$\int_{1}^{t}Z_{s}(z_{s})ds$')
#         ax.set_xlabel(r'$t$')

#         ax = axs[1, 1]
#         ax.plot(t_intpnts.squeeze().cpu().detach().numpy(), (1/integ_t_intpnts).cpu().detach().numpy(), 'b-')
#         ax.set_ylabel(r'$\frac{1}{\int_{1}^{t}Z_{s}(z_{s})ds}$')
#         ax.set_xlabel(r'$t$')
        
#         plt.show()

# def generate(self, batch_size=100, max_num_time_steps=100, random_seed=24, y=None):
    #     """
    #     Generate 'novel' points by sampling from p(z_1) and then using ancestral 
    #     sampling (backwards in time) to obtain 'novel' \hat{z}_0=\hat{x}.

    #     Args:
    #         batch_size (int): Number of 'novel' points '\hat{x}' to generate.
    #             (Default: 100)
    #         max_num_time_steps (int): Maximal number of ancestral sampling (time) 
    #             steps to use for backward propagation through time.
    #             (Default: 100)
    #         random_seed (int): Random seed to be used for all the sampling in
    #             the generation process.
    #             (Default: 24)
    #         y (int or None): Conditional class to guide to.
        
    #     Return:
    #         (torch.tensor): 'Novel' points as 2D torch tensor \hat{x} of shape
    #             (batch_size, #x-features) generated in backward diffusion.
        
    #     """
    #     # Parse y if it is not None
    #     if y is not None:
    #         y = y*torch.ones(batch_size, dtype=torch.int).reshape(-1, 1)

    #     # Set a random seed
    #     self.set_random_seed(random_seed)

    #     # Initialize the time of each point in the batch to 1
    #     # Remark: 
    #     batch_t = torch.ones(batch_size, 1)
    #     if self._debug:
    #         print(f"batch_t.shape: {batch_t.shape}")
    #         print(f"batch_t[:10]:  {batch_t[:10]}")

    #     # Sample z_(t=1)=z_1 and set it as the initial z_t
    #     batch_z_t = self._sample_z_1(batch_size)
    #     #if self._debug:
    #     print(f"batch_z_1.shape: {batch_z_t.shape}")
    #     print(f"batch_z_1[:10]:  {batch_z_t[:10]}")

    #     # Loop over the time steps
    #     for step in range(max_num_time_steps):
    #         ## A) Sample the next time for each point in the batch
    #         # A1) Determine the backward transition rate vector \hat{R}_{t}(z_t, :) for each point in the batch
    #         batch_hat_R_t_z_t_vec = self._predict_hat_R_t_z_start_t_vec(batch_z_t, batch_t, train_or_eval='eval')
    #         #if self._debug:
    #         print(f"batch_hat_R_t_z_t_vec.shape: {batch_hat_R_t_z_t_vec.shape}")
    #         print(f"batch_hat_R_t_z_t_vec[:10]:  {batch_hat_R_t_z_t_vec[:10]}")

    #         # A2) Determine \hat{Z}_t(z_t)=\sum_{z!=z_t} \hat{R}_t(z_t, z) that corresponds 
    #         #    to the sum over the backward rates of all transitions that jump out of the 
    #         #    current z_t (i.e. the total rate to transition/jump out of the current z_t)
    #         batch_hat_Z_t_z_t = self._get_Z_t_z_t(batch_z_t, batch_hat_R_t_z_t_vec)
    #         #if self._debug:
    #         print(f"batch_hat_Z_t_z_t.shape: {batch_hat_Z_t_z_t.shape}")
    #         print(f"batch_hat_Z_t_z_t[:10]:  {batch_hat_Z_t_z_t[:10]}")

    #         # A3) Draw i.i.d. uniform random variables
    #         batch_u = torch.rand(batch_size)
    #         #if self._debug:
    #         print(f"batch_u.shape: {batch_u.shape}")
    #         print(f"batch_u[:10]:  {batch_u[:10]}")

    #         # A4) Determine the holding time for each point in the batch using the
    #         #    corresponding formula in the Guillespi algorithm
    #         #    \tau^{holding} = log(1/u)/[\sum_{z} \hat{R}_t{z_t, z)]
    #         #    for rates that are constant in time (approximation here).
    #         #    As we are only considering jump transitions here (i.e. transitions
    #         #    to states different from the initial state), the denominator becomes 
    #         #    \sum_{z!=z_t} \hat{R}_t{z_t, z)=\hat{Z}_t(z_t).
    #         #    Remark: Use log(1/u) = -log(u)
    #         batch_t_holding = -torch.log(batch_u)/(batch_hat_Z_t_z_t+self._eps)
    #         #if self._debug:
    #         print(f"batch_t_holding.shape: {batch_t_holding.shape}")
    #         print(f"batch_t_holding[:10]:  {batch_t_holding[:10]}")

    #         # A5) Determine the next time for each point in the batch
    #         # A5a) Add the holding time to all the current times to obtain the next times
    #         batch_t = batch_t - batch_t_holding.reshape(-1, 1)
    #         #if self._debug:
    #         print(f"[raw] batch_t.shape: {batch_t.shape}")
    #         print(f"[raw] batch_t[:10]:  {batch_t[:10]}")

    #         # A5b) Enforce a minimal time of t=0 for all points in the batch
    #         batch_zeros = torch.zeros_like(batch_t)
    #         batch_t     = torch.maximum(batch_t, batch_zeros)
    #         #if self._debug:
    #         print(f"[corrected] batch_t.shape: {batch_t.shape}")
    #         print(f"[corrected] batch_t[:10]:  {batch_t[:10]}")

            
    #         ## B) After holding, jump to another state for this new time
    #         # B1) Determine the backward transition rate vector \hat{R}_{t}(z_t, :) for each point in the batch
    #         batch_hat_R_t_z_t_vec = self._predict_hat_R_t_z_start_t_vec(batch_z_t, batch_t, train_or_eval='eval')
    #         #if self._debug:
    #         print(f"batch_hat_R_t_z_t_vec.shape: {batch_hat_R_t_z_t_vec.shape}")
    #         print(f"batch_hat_R_t_z_t_vec[:10]:  {batch_hat_R_t_z_t_vec[:10]}")

    #         # B2) Determine the probability vector to jumpt to a new state for each point in the batch
    #         prob_vec_jump_z_t = self._get_jump_prob_vec(batch_z_t, batch_hat_R_t_z_t_vec)

    #         # B3) Sample z^{jump}_t from a categorical distribution with the probability vector p^{jump}_t[z_t]
    #         #     that corresponds to p^{jump}_t(z^{jump}_t|z_t).
    #         q_z_jump_t_z_t = torch.distributions.categorical.Categorical(prob_vec_jump_z_t) 
    #         batch_z_jump_t = q_z_jump_t_z_t.sample()
    #         #if self._debug:
    #         print(f"batch_z_t.shape: {batch_z_t.shape}")
    #         print(f"batch_z_t[:10]:  {batch_z_t[:10]}")
    #         print(f"batch_z_jump_t.shape: {batch_z_jump_t.shape}")
    #         print(f"batch_z_jump_t[:10]:  {batch_z_jump_t[:10]}")
    #         print()

    #         # B4) Update batch_z_t with the values in batch_z_jump_t for all the points that haven't reached t=0 yet
    #         ix            = torch.where(batch_t.squeeze()>0)
    #         batch_z_t[ix] = batch_z_jump_t[ix]
    #         #if self._debug:
    #         print(f"ix[:10]:  {ix[:10]}")
    #         print(f"[updated] batch_z_t.shape: {batch_z_t.shape}")
    #         print(f"[updated] batch_z_t[:10]:  {batch_z_t[:10]}")
    #         print()

    #         # If all the batch times are 0, we can stop early
    #         if torch.all(batch_t==0):
    #             print(f"Propagated to all t=0 for all points in the batch after {step} timesteps, thus stopping early.")
    #             break

    #     # Return the final z_t
    #     return batch_z_t