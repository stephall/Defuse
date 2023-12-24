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

class DiscreteDiffusionManager(base_diffusion_manager.BaseDiffusionManager):
    """
    Define a manager to handle continuous time diffusion in discrete 1D spaces.
    """
    # Define a tiny number to be used for numerical stability (i.e. when dividing or taking logarithms)
    _eps = 1e-10

    def __init__(self, *args, prob_vec_1_list=None, **kwargs):
        """
        
        """
        # Initialize the base class
        super().__init__(*args, **kwargs)

        ###################################################################################################################################################################################################################################
        ### Parse input 'prob_vec_1'
        ###################################################################################################################################################################################################################################
        if prob_vec_1_list is None:
            err_msg = f"For discrete diffusion the input 'prob_vec_1_list' -- that specifies the probability vectors of each component of the fully noised state z_1 sampled from the distribution 'p_1(z_1)=\Prod_(c=1)^(discrete-space-dim) Cat(z^c_1|p^c_vec_1)' -- must be passed during initialization of the diffusion manager object."
            raise ValueError(err_msg)
        
        # Loop over the probability vectors in prob_vec_1 and preprocess them 
        # before adding them to the class attribute 'prob_vec_1_list'
        self.cardinality_list          = list()
        self.encoding_start_index_list = list()
        _prob_vec_1_list               = list()
        A_c_1_list                     = list()
        start_index = 0 # Initialize the starting index to 0
        for d, prob_vec_1 in enumerate(prob_vec_1_list):
            if self._debug:
                print(f"Component: {d}")
            
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
            
            # Normalize prob_vec_1 and append it to the corresponding list
            # while reshaping it to shape (-1, 1) so that the list entries 
            # can be concatenated below
            prob_vec_1 = prob_vec_1/torch.sum(prob_vec_1)
            _prob_vec_1_list.append(prob_vec_1.reshape(-1, 1))

            # Display when debugging
            if self._debug:
                print(f"prob_vec_1.shape: {prob_vec_1.shape}")
                print(f"prob_vec_1: {prob_vec_1}")

            # Get the cardinality of the discrete space of the current component
            # 'c' from 'prob_vec_1' that is of shape (cardinality_c,) and append,
            # where cardinality_c is the cardinality of the c-th discrete subspace,
            # it to the corresponding list
            cardinality_c = prob_vec_1.shape[0]
            self.cardinality_list.append(cardinality_c)

            # Append the current starting index for the segment in the x-encoding 
            # corresponding to the current component to the correspondibg list.
            # Example:
            #   Consider a 2 dimensional x space with cardinalities 2 and 3.
            #   The corresponding encoded space will be 5 dimensional and for example 
            #   x=[0, 0] is mapped to x_encoded=[1, 0, 1, 0, 0]
            #   x=[1, 2] is mapped to x_encoded=[0, 1, 0, 0, 1]
            #   Thus, the segment in the encoded vector corresponding to the first 
            #   component of x (c=0) starts at index 0 and the segment corresponding to
            #   the second component of x (c=1) starts at index 2. 
            self.encoding_start_index_list.append(start_index)

            # Update the starting index by adding the cardinality of the current component
            start_index += cardinality_c

            ###################################################################################################################################################################################################################################
            # The base rate matrix R^c_b = Identity-A^c_1 of component c is specified over the matrix
            # A^c_1=1_vec*prob_vec^c_1^T where prob_vec^c_1 is the probability vector intended 
            # for t=1 for component d and 1_vec is a vector full of 1s [i.e. 1_vec^T=(1, ..., 1)] 
            # of size equal to cardinality_c.
            A_c_1 = torch.ones(cardinality_c, 1)*prob_vec_1.reshape(1, -1)
            A_c_1_list.append(A_c_1)

            # Ensure that the sum over each of the rows of R^c_b is 0
            # Remark: This is a condition for the rate matrix R^c_t, and because
            #         we use R^c_t = gamma'(t)R^c_b, this condition translates to a 
            #         condition for R^c_b.
            R_c_b = A_c_1 - torch.eye(cardinality_c)
            if torch.any(1e-5<torch.abs(torch.sum(R_c_b, dim=1))):
                err_msg = f"The rows of the (base) rate matrix must sum to 0, which is not the case for the base rate matrix\nR^c_b={R_c_b} for component c={c}."
                raise ValueError(err_msg)
            
        # Construct the probability vector for t=1
        self.prob_vec_1 = torch.cat(_prob_vec_1_list).squeeze()

        # Construct A_1 as block diagonal matrix in the basis of the encoded x-space
        # where the blocks correspond to the A_1 matrices for each component in x.
        # Remark: torch.block_diag expects each matrix intended as block in the to be
        #         created matrix as input argument. Thus, we have to updack the list
        #         containing the A^c_1 matrices of each component 'c'.
        A_1 = torch.block_diag(*A_c_1_list)

        # Also construct the identity matrix in the same space (i.e. the encoded x-space)
        Id  = torch.eye(self.x_encoding_space_dim)

        # Make the tensors (i.e. matrices) A_1 and I of shape 
        # (self.x_encoding_space_dim, self.x_encoding_space_dim)
        # tensors of shape (1, self.x_encoding_space_dim, self.x_encoding_space_dim).
        self.A_1 = torch.unsqueeze(A_1, dim=0)
        self.Id  = torch.unsqueeze(Id, dim=0)
        self.R_b = self.A_1-self.Id

        # Define a subspace projection matrix that can be used to obtain a mask
        # corresponding to the component
        # Example: Consider a 2D categorical space with cardinalities 2 and 3
        #          Using the subspace projection matrix 
        #          component_subspace_projector=
        #          (1 1 0 0 0)
        #          (1 1 0 0 0)
        #          (0 0 1 1 1)
        #          (0 0 1 1 1)
        #          (0 0 1 1 1)
        #          any component-wise one-hot encoded vector can be mapped
        #          to a mask-vector that has 1s in the segments corresponding
        #          that had a 1.
        #          E.g. (1 0 1 0 0)@subspace_proj = (1 1 1 1 1)
        #          or   (0 0 1 0 0)@subspace_proj = (0 0 1 1 1)
        #          or   (0 0 0 1 0)@subspace_proj = (0 0 1 1 1)
        #          or   (0 0 0 0 1)@subspace_proj = (0 0 1 1 1)
        #          or   (1 0 0 0 0)@subspace_proj = (1 1 0 0 0)
        #          or   (0 1 0 0 0)@subspace_proj = (1 1 0 0 0)
        self._component_subspace_projector = torch.block_diag(*[torch.ones(cardinality, cardinality) for cardinality in self.cardinality_list])

        # Initialize the class attribute corresponding to the temperature used
        # when using guided-sampling for the property(-guide) distribution to 1
        self._guide_temp = 1

        # Define the softmax function that should be applied along the
        # second axis (i.e. the feature axis)
        self.softmax_fn = torch.nn.Softmax(dim=1)

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

    @property
    def x_space_dim(self):
        """ Return the dimension of the discrete state space of x that is equivalent to the length of self.cardinality_list. """
        return len(self.cardinality_list)
    
    @property
    def x_encoding_space_dim(self):
        """ Return the dimension of the space of the encoded x that is equivalent to the sum of all cardinalities within self.cardinality_list. """
        return int(sum(self.cardinality_list))
    
    def encode_x(self, x):
        """ 
        Encode the input tensor x.
        
        Args:
            x (torch.tensor): To be encoded 2D torch tensor of shape (batch_size, self.x_space_dim).
        
        Return:
            (torch.tensor): Encoded x as 2D torch tensor of shape (batch_size, self.x_encoding_space_dim).
        
        """
        # Create a list containing each encoded component 'c' of x, that each is a 2D torch tensors of 
        # shape (batch_size, self.cardinality_list[c]), and concatenate them along the encoding 
        # (i.e. second) axis for each point in the batch.
        return torch.cat([self.encode_x_c(x, c) for c in range(self.x_space_dim)], dim=1)
    
    def encode_x_c(self, x, c):
        """
        Encode the c-th component of x. 
        
        Args:
            x (torch.tensor): 2D torch tensor of shape (batch_size, self.x_space_dim).
            c (int): Component to be encoded in x.

        Return:
            (torch.tensor): Encoded c-th component of x as 2D torch tensor of shape 
                (batch_size, self.cardinality_list[c])
        
        """
        # Define a one-hot encoding for the c-th component of the categorical vector
        # x using the cardinality of the c-th component of the discrete space
        one_hot_encoding = encodings.OneHotEncoding1D(dim=self.cardinality_list[c])

        # Return the one-hot encoded c-th component 
        return one_hot_encoding(x[:, c])

    
    def decode_x(self, x_encoded):
        """ 
        Decode the encoded input tensor x.
        
        Args:
            x_encoded (torch.tensor): 2D torch tensor of shape (batch_size, self.x_encoding_space_dim).
        
        Return:
            (torch.tensor): Decoded x as 2D torch tensor of shape (batch_size, self.x_space_dim).
        
        """
        # Create a list containing each decoded component 'c' of x, that each is a 2D torch tensors 
        # of shape (batch_size, 1), and concatenate them along the encoding (i.e. second) axis for 
        # each point in the batch.
        return torch.cat([self.decode_x_c(x_encoded, c) for c in range(self.x_space_dim)], dim=1)
    
    def decode_x_c(self, x_encoded, c):
        """ 
        Decode the c-th component of an encoded x. 
        
        Args:
            x_encoded (torch.tensor): 2D torch tensor of shape (batch_size, self.x_encoding_space_dim).
            c (int): Component to be decoded in x.

        Return:
            (torch.tensor): Encoded c-th component of x as 2D torch tensor of shape 
                (batch_size, 1)
        
        """
        # Determine the (batched) component-vector within the subspace of the c-th discrete 
        # component of the (batched) encoded vector x_encoded
        x_c_encoded = self._get_component_vector(x_encoded, c)

        # Because each component of x is one-hot encoded, the categorical entry 'x_c'
        # of the c-th component of x can be obtained as the argmax of the
        # encoded component x_c_encoded over the feature (i.e. second axis), e.g. 
        #   x_c_encoded=[1, 0, 0] <=> x_c=0
        #   x_c_encoded=[0, 1, 0] <=> x_c=1
        #   x_c_encoded=[0, 0, 1] <=> x_c=2
        return torch.argmax(x_c_encoded, dim=1).reshape(-1, 1)
    
    def _get_component_vector(self, batched_vector, c):
        """
        Return the vector in the subspace of the c-th component of the input vector.

        Example: Consider a 2D discrete state space with cardinalities [2, 3] and
                 let v=(v1, v2, v3, v4, v5). 
                 The 0-th component vector of v is given by v^0 = (v1, v2) and
                 the 1-th component vector of v is given by v^1 = (v3, v4, v5).

        Args:
            batched_vector (torch.tensor): (Batched) vector of which the vector
                within the subspace of the c-th component should be returned as 2D
                torch tensor of shape (batch_size, self.x_encoding_space_dim).
            c (int): Component of the discrete space whose corresponding vector within
                the subspace of the encoded space should be returned.
            
        Return:
            (torch.tensor): (Batched) vector within the subspace of the c-th component
                of batched_vector as 2D torch tensor of shape (batch_size, cardinality_c).                

        """
        # Determine the starting and end indices for the segments (i.e. subspace) in the 
        # encoded space corresponding to the c-th component in the discrete space.
        # The start index can be looked up in 'self.encoding_start_index_list' and the
        # end index is the start index plus the cardinality of the current component
        c_start_index = self.encoding_start_index_list[c]
        c_end_index   = c_start_index + self.cardinality_list[c]

        # Return the sliced vector that correspond to the subspace of the c-th component
        # of the discrete space
        return batched_vector[:, c_start_index:c_end_index]
    
    def Q(self, batch_t):
        """
        Return the batched (block diagonal) transition matrice Q_t whose blocks specify the noising 
        distribution of each component z^c_t of z_t:
        p_t(z^c_t) = Cat(z^c_t|prob_vec^c_t=enc(x^c)*Q^c_t).

        Remark: Different points in the batch can have different times.
                So the batch (i.e. first) axis takes also the role of
                a 'time axis'.

        Args:
            batch_t (torch.tensor): (Batched) time as torch tensor of shape (batch_size, 1).
        
        Return:
            (torch.tensor): (Batched) rate matrix for the time of each point in the batch as 3D
                torch tensor of shape (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)

        """
        # Determine kappa(t) that is of shape (batch_size, 1)
        batch_kappa_t = torch.sqrt(self.kappa2(batch_t))

        # Unsqueeze kappa(t) to shape (batch_size, 1, 1)
        batch_kappa_t = torch.unsqueeze(batch_kappa_t, dim=2)

        # Return Q_t = kappa(t)*Id+[1-kappa(t)]*A_1 as 3D torch tensor of shape 
        # (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)
        # Remark: self.Id and self.A_1 are both of shape (1, self.x_encoding_space_dim, self.x_encoding_space_dim)
        batch_Q_t = batch_kappa_t*self.Id + (1-batch_kappa_t)*self.A_1
        return batch_Q_t.to_sparse()
    
    def R(self, batch_t):
        """
        Return the batched rate matrix R_t=R_b*gamma'(t).
        
        Args:
            batch_t (torch.tensor): (Batched) time as torch tensor of shape (batch_size, 1).
        
        Return:
            (torch.tensor): (Batched) rate matrix for the time of each point in the batch as 3D
                torch tensor of shape (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)
        """
        # Determine gamma'(t) that is of shape (batch_size, 1)
        batch_deriv_gamma_t = self.deriv_gamma(batch_t)

        # Unsqueeze gamma'(t) to shape (batch_size, 1, 1)
        batch_deriv_gamma_t = torch.unsqueeze(batch_deriv_gamma_t, dim=2)

        # Return R_t = R_b*gamma'(t) as 3D torch tensor of shape 
        # (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)
        # Remark: self.R_b is of shape (1, self.x_encoding_space_dim, self.x_encoding_space_dim)
        batch_R_t = batch_deriv_gamma_t*self.R_b
        return batch_R_t.to_sparse()

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
        # (batch_size, self.x_space_dim)
        batch_size = batch_x.shape[0]

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
        # that will be a 2D torch tensor of shape (batch_size, dim_jump_state_space[z_t])
        # where jump_state_space[z_t] is the dimension of the states that can be reached
        # with a jump in a single component of z_t and its dimension corresponds to the
        # number of all of these states.
        batch_R_t_z_t_vec = self._get_R_t_z_start_t_vec(batch_z_t, batch_t)

        # Determine the (batched) forward normalizing constant Z_t(z_t)
        # that will be a 1D torch tensor of shape (batch_size,)
        batch_Z_t_z_t = self._get_Z_t_z_t(batch_z_t, batch_R_t_z_t_vec)

        # Determine the (batched) probability vector to jump from z_t to any other state
        # that differs from z_t in only a single component at time t. 
        # This probability vector will be a 2D torch tensor of shape (batch_size, self.x_enc_dim).
        # Remark: The probabilities are over the encoded x-space, i.e. they specify what 
        #         the probability is to change to a new specific entry in encoded(z_t). 
        # Example: Let's consider a 2D x-space with cardinalities 2 and 3.
        #          Assume z_t=[0, 1] with encoded(z_t)=[1, 0, 0, 1, 0].
        #          Because a jump-transition must occur by construction and only happens 
        #          in a single component of z_t, the probabilities to change the entries that 
        #          are already 1, are zero,
        #          i.e. prob_vec^jump[z_t] = [0, p1, p2, 0, p4] while p2+p3+p5=1.
        #          Imagine we sample w.r.t. this probability vector and obtain a change at
        #          'jump' index 2 (sampled with prob. p2) that can be represented as [0, 0, 1, 0, 0].
        #          This means that the state encoded(z_t)=[1, 0, 0, 1, 0] will jump to the new
        #          state encoded(z^jump_t)=[1, 0, 1, 0, 0].
        prob_vec_jump_t_z_t = self._get_jump_prob_vec(batch_z_t, batch_R_t_z_t_vec)

        # Sample the 'jump index' from a categorical distribution with the probability vector 
        # p^{jump}_t[z_t] that corresponds to p^{jump}_t(jump_index|z_t) where the jump_index
        # is an integer in [0, self.x_encoding_space_dim-1] and specifies the index in the encoded
        # x space that will be the new state of the corresponding component.
        q_jump_t_index_z_t = torch.distributions.categorical.Categorical(prob_vec_jump_t_z_t) 
        batch_jump_index_t = q_jump_t_index_z_t.sample()
        if self._debug:
            print(f"batch_z_t.shape: {batch_z_t.shape}")
            print(f"batch_z_t[:10]:  {batch_z_t[:10]}")
            print()
            print(f"batch_jump_index_t.shape: {batch_jump_index_t.shape}")
            print(f"batch_jump_index_t[:10]:  {batch_jump_index_t[:10]}")
            print()

        # Perform a transition from state z_t to state z^jump_t specified by the transition index 'batch_jump_index_t'
        batch_z_jump_t = self._transition_to_new_state(batch_z_t, batch_jump_index_t)

        # Determine \hat{R}(z^{start}_t, :) where the starting state z^{start}_t is z^jump_t here
        # as 2D tensor of shape (batch_size, self.x_enc_dim) corresponding to the inverse (i.e. back
        # ward in time) transition rates from start state z^{start}_t to any other end state z^{end}_t
        batch_hat_R_t_z_jump_t_vec = self._predict_hat_R_t_z_start_t_vec(batch_z_jump_t, batch_t, train_or_eval='train', batch_y=None)

        # Determine the (batched) encoded jump index for the transition from z^jump_t to z_t, which corresponds 
        # to the inverse of the transition from z_t to z^jump_t, that will be a 2D torch tensor of shape
        # (batch_size, self.x_encoding_space_dim)
        batch_inverse_jump_index_t_encoded = self._get_encoded_inverse_jump_index(batch_z_t, batch_jump_index_t)

        # Determine R_t(z^jump_t, z_t)=R_t(z^jump_t, :)@encoded(transition_index(z^jump_t->z_t))
        batch_hat_R_t_z_jump_t_z_t = utils.batched_matmul(batch_hat_R_t_z_jump_t_vec, batch_inverse_jump_index_t_encoded)
        if self._debug:
            print(f"batch_hat_R_t_z_jump_t_z_t.shape: {batch_hat_R_t_z_jump_t_z_t.shape}")
            print(f"batch_hat_R_t_z_jump_t_z_t[:10]:  {batch_hat_R_t_z_jump_t_z_t[:10]}")
            print()

        # Determine the (batched) backward normalizing constant \hat{Z}_t(z^jump_t)
        # that will be a 1D torch tensor of shape (batch_size,)
        batch_hat_Z_t_z_jump_t = self._get_Z_t_z_t(batch_z_jump_t, batch_hat_R_t_z_jump_t_vec)
        if self._debug:
            print(f"batch_hat_Z_t_z_jump_t.shape: {batch_hat_Z_t_z_jump_t.shape}")
            print(f"batch_hat_Z_t_z_jump_t[:10]:  {batch_hat_Z_t_z_jump_t[:10]}")
            print()

        # Calculate the loss function
        # \hat{Z}_t(z^jump_t) - Z_t(z_t)*log[\hat{R}^theta_t(z^jump_t, z_t)]
        # for each point in the batch (i.e. a the result is a loss vector over the batch)
        # Remark: Add a tiny value (self._eps) to the logarithm input for numerical stability.
        batch_loss_t = batch_hat_Z_t_z_jump_t - batch_Z_t_z_t*torch.log(batch_hat_R_t_z_jump_t_z_t+self._eps)
        if self._debug:
            print(f"batch_loss_t.shape: {batch_loss_t.shape}")
            print(f"batch_loss_t[:10]:  {batch_loss_t[:10]}")
            print()

        # Get the (batched) time-dependent weight for the loss, multiply this weight to
        # the previously determined loss and sum over all the losses of the batch points
        batch_time_weight_t = self.time_weight(batch_t)
        loss_t              = torch.sum(batch_time_weight_t*batch_loss_t)
        if self._debug:
            print(f"loss_t.shape: {loss_t.shape}")
            print(f"loss_t:       {loss_t}")
            print()
    
        return loss_t
    
    def _get_jump_prob_vec(self, batch_z_start_t, batch_R_t_z_start_t_vec):
        """
        Return the jump probabilities from a state z^{start}_t to any other
        state z^{end}_t!=z^{start}_t.
        
        Remark: This method works both for forward or backward time rates!

        Args:
            batch_z_start_t (torch.tensor): The batched start state to jump 
                from as 2D torch tensor of shape (batch_size, self.x_space_dim).
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
    
    def _transition_to_new_state(self, batch_z_start, batch_transit_index):
        """
        Perform a transition from initial state z^start to a final state 
        z^end specified by the transition index.

        Remarks: (1) We only allow transitions that lead to a change in a single
                     component (and not in multiple components).
                     Thus, the initial state z^start and final state z^end
                     will either be the same (identity-transition) or differ in
                     only a single component (jump-transition).
                 (2) The transition index specifies the index at which the entry
                     in z^end should be set to 1. If the corresponding entry
                     in z^start was already 1, the transition is an identity-transition,
                     and if it was 0, the transition is a jump-transition.
                     Example: Consider a 2D discrete space of cardinality [2, 3].
                              Let z^start=[0, 1] <=> encoded(z^start)=[1, 0, 0, 1, 0]
                              There are 5 [=dim(encoded_space)] possible transitions:
                                (i)   transition_index=0: encoded(z^end)=[1, 0, 0, 1, 0] (identity-transition in first component)
                                (ii)  transition_index=1: encoded(z^end)=[0, 1, 0, 1, 0] (jump-transition in first component)
                                (iii) transition_index=2: encoded(z^end)=[1, 0, 1, 0, 0] (jump-transition in second component)
                                (iv)  transition_index=3: encoded(z^end)=[1, 0, 0, 1, 0] (identity-transition in second component)
                                (v)   transition_index=4: encoded(z^end)=[1, 0, 0, 0, 1] (jump-transition in second component)
                              By construction there will be 'dim(discrete_space)' identity-transitions (one in each component of the discrete space)
                              and thus 'dim(encoded_space)-dim(discrete_space)' jump-transitions (i.e. cardinality[c]-1 jump-transitions in each component c).
        
        Args:
            batch_z_start (torch.tensor): (Batched) initial state of the transition
                as 2D torch tensor of shape (batch_size, self.x_encoding_space_dim).
            batch_transition_index (torch.tensor): (Batched) transition indices
                (i.e. integers in [0, self.x_encoding_space_dim-1]) as 1D torch
                torch tensor of shape (batch_size,).
        
        Return:
            (torch.tensor): (Batched) final state z^end of the transition as 2D torch
                tensor of shape (batch_size, self.x_encoding_space_dim).

        """
        # One-hot encode the (batched) transition index, i.e. the transition index of each
        # point in the batch, as vector within the encoded x-space
        one_hot_encoding = encodings.OneHotEncoding1D(dim=self.x_encoding_space_dim)
        batch_transit_index_encoded = one_hot_encoding(batch_transit_index)
        if self._debug:
            print(f"batch_transit_index_encoded.shape: {batch_transit_index_encoded.shape}")
            print(f"batch_transit_index_encoded[:10]:  {batch_transit_index_encoded[:10]}")
            print()

        # Get a mask with 1s for the entries of the component-subspace within which
        # the transition occurs, and 0s in all the other component subspaces.
        # To obtain this mask, left-multiply the encoded transition index with the
        # component subspace projector matrix for this purpose.
        # Example: Consider a 2D categorical space with cardinalities 2 and 3.
        #          transit_index=2, so that one-hot-encoded(transit_index)=[0, 0, 1, 0, 0].
        #          Thus, [0, 0, 1, 0, 0]@component_subspace_projector = [0, 0, 1, 1, 1]
        #          that is a mask within the encoded space representing any part of
        #          an encoded vector corresponding to the component the jump should
        #          occur in, which is the second component here.
        batch_transit_component_mask = batch_transit_index_encoded@self._component_subspace_projector

        # Determine the final state z^end
        # encoded(z^end) = encoded(z^start) + one-hot-encoded(transit_index) - transit_component_mask*encoded(z^start)
        #                = (1-transit_component_mask)*encoded(z^start) + one-hot-encoded(transit_index)
        # Example: Consider a 2D categorical space with cardinalities 2 and 3.
        #          transit_index=2, so that one-hot-encoded(transit_index)=[0, 0, 1, 0, 0]
        #          leading to the transition component mask [0, 0, 1, 1, 1].
        #          Assume z^start=[0, 1] with encoded(z^start)=[1, 0, 0, 1, 0] so that
        #          multiplication with the transition component mask results in
        #          [1, 0, 0, 1, 0]*[0, 0, 1, 1, 1]=[0, 0, 0, 1, 0]
        #          corresponding to the one-hot-encoded 'initial entry' within z^start before the
        #          transition that will be changed to a 'final entry' after the transition.
        #          By add the one-hot-encoded(transit_index) and subtracting this one-hot-encoded 
        #          'initial' entry from z^start one can obtain z^end, i.e. for the example
        #          encoded(z^end) = encoded(z^start) + one-hot-encoded(transit_index) - transit_component_mask*encoded(z^start)
        #                         = [1, 0, 0, 1, 0] + [0, 0, 1, 0, 0] - [0, 0, 0, 1, 0]
        #                         = [1, 0, 1, 0, 0]
        #          That can be decoded to z^end=[0, 0].
        batch_z_start_encoded = self.encode_x(batch_z_start)
        batch_z_end_encoded   = (1-batch_transit_component_mask)*batch_z_start_encoded + batch_transit_index_encoded
        if self._debug:
            print(f"batch_z_start_encoded.shape: {batch_z_start_encoded.shape}")
            print(f"batch_z_start_encoded[:10]:  {batch_z_start_encoded[:10]}")
            print(f"batch_z_end_encoded.shape:   {batch_z_end_encoded.shape}")
            print(f"batch_z_end_encoded[:10]:    {batch_z_end_encoded[:10]}")
            print()

        # Decode z^end from encoded(z^end)
        batch_z_end = self.decode_x(batch_z_end_encoded)
        if self._debug:
            print(f"batch_z_start.shape: {batch_z_start.shape}")
            print(f"batch_z_start[:10]:  {batch_z_start[:10]}")
            print(f"batch_z_end.shape:   {batch_z_end.shape}")
            print(f"batch_z_end[:10]:    {batch_z_end[:10]}")
            print()

        return batch_z_end
    
    def _get_encoded_inverse_jump_index(self, batch_z_start, batch_jump_index):
        """
        Return the encoded jump index for the jump-transition z^{end} to z^{start} that 
        corresponds to the inverse of the jump-transition from z^{start} to z^{end}, which
        is specified by the input z^{start} and the transition index jump_index.

        Args:
            batch_z_start (torch.tensor): (Batched) start state z^{start} as 2D torch tensor
                of shape (batch_size, self.x_space_dim).
            batch_jump_index (int): (Batched) transition index specifying the transition from z
                to some other state z^end.

        Return:
            (torch.tensor): (Batched) encoded inverse jump index as 2D torch tensor of
                shape (batch_size, self.x_encoding_space_dim).
        
        Example:
            Consider a 2D discrete state space with cardinalities [2, 3].
            Let z^start=[0, 1] <=> encoded(z^start)=[1, 0, 1, 0, 0] and
            assume the transition index jump_index=3 s.t. 
            encoded(z^end)=[1, 0, 0, 1, 0].
            The inverse jump-transition from z^end to z^start is specified
            by the transition index inverse_jump_index=2.

            We can obtain the inverse jump index from the start state z^{start} and
            the jump index in the following way.
            Step 1: 
            Obtain a mask that has 1s for the discrete component in which the
            jump-transition happens by calculating
            component_mask = encoded(jump_index)@self._component_subspace_projector
                           = [0, 0, 0, 1, 0]@self._component_subspace_projector
                           = [0, 0, 1, 1, 1]
            
            Step 2:
            Use elementwise multiplication (i.e. boolean AND operation) of the component
            mask and the encoded start state z^start to obtain the entry of the transition
            component in the encoded z^start that was 1 before the transition (and zero
            afterwards). This entry corresponds to the encoded inverse jump index.
            encoded(inverse_jump_index) = component_mask*encoded(z_start)
                                        = [0, 0, 1, 1, 1]*[1, 0, 1, 0, 0]
                                        = [0, 0, 1, 0, 0]

        """
        # One-hot encode the jump index as vector within the encoded x-space and also
        # encode the start space z^{start}.
        one_hot_encoding = encodings.OneHotEncoding1D(dim=self.x_encoding_space_dim)
        batch_jump_index_encoded = one_hot_encoding(batch_jump_index)
        batch_z_t_encoded = self.encode_x(batch_z_start)

        if self._debug:
            print(f"batch_jump_index_encoded.shape: {batch_jump_index_encoded.shape}")
            print(f"batch_jump_index_encoded[:10]:  {batch_jump_index_encoded[:10]}")
            print()

        # Determine the component mask (see Step 1 in example within docstring above)
        batch_jump_component_mask = batch_jump_index_encoded@self._component_subspace_projector

        # Determine the encoded inverse jump index (see Step 2 in example within docstring above)
        batch_inverse_jump_index_encoded = batch_jump_component_mask*batch_z_t_encoded
        if self._debug:
            print(f"batch_inverse_jump_index_encoded.shape: {batch_inverse_jump_index_encoded.shape}")
            print(f"batch_inverse_jump_index_encoded[:10]:  {batch_inverse_jump_index_encoded[:10]}")
            print()

        return batch_inverse_jump_index_encoded
    
    def _get_Z_t_z_t(self, batch_z_t, batch_R_t_z_t_vec):
        """
        Return the total rate to transition out from (i.e. through the allowed jump-transition) 
        a state z_t to any other state z at time t, i.e. Z_t(z_t) = \sum_{z!=z_t} R_t(z_t, z).
        The time t is not explicitly used within this method but implicitly contained within
        the inputs z_t and R_t(z_t, :)

        Remark: This method works both for forward or backward time rates!

        Args:
            batch_z_t (torch.tensor): The batched state from which to jump 
                from as 2D torch tensor of shape (batch_size, self.x_space_dim).
            batch_R_t_z_t_vec (torch.tensor): The transition rates R_t(z_t, z) 
                at time t from z_t to any state z represented as vector 
                R_t(z_t, :) for each point in the batch in the form of a 2D torch 
                tensor of shape (batch_size, self.x_encoding_space_dim).

        Return:
            (torch.tensor): The total jump rate to transition out of state 'z_t' for 
                each point in the batch as 1D tensor of shape (batch_size,).

        """
        # Encode z_t resulting in a 2D torch tensor of shape (batch_size, self.x_encoding_space_dim)
        batch_z_t_encoded = self.encode_x(batch_z_t)

        # First, determine R_t(z_t, z_t).
        # We know that for any state z, R_t(z, z_t) corresponds to R_t(z, :)@encoded(z_t) 
        # (where @ is matrix multiplication) because encoded(z_t)=direct_sum_{c} one-hot-encoded(z^c_t) 
        # and R_t(z, :)=direct_sum_{c} R^c_t(z, :) (because the full R_t is block diagonal and thus 
        # R^c corresponds to the c-th component) s.t. 
        # R_t(z, z_t) = direct_sum_{c} R^c_t(z, :)@one-hot-encoded(z^c_t) = R_t(z, :)@encoded(z_t).
        batch_R_t_z_t_z_t = utils.batched_matmul(batch_R_t_z_t_vec, batch_z_t_encoded)
        if self._debug:
            print(f"batch_R_t_z_t_z_t.shape: {batch_R_t_z_t_z_t.shape}")
            print(f"batch_R_t_z_t_z_t[:10]:  {batch_R_t_z_t_z_t[:10]}")

        # Second, determine the normalizing constant Z_t(z_t) = sum_{z!=z_{t}}R_t(z_t, z) 
        # by using the property of R_t that its rows must sum to 0, 
        # i.e. sum_{z}R_t(z_t, z), s.t. Z_t(z_t) = -R_t(z_t, z_t).
        batch_Z_t_z_t = -batch_R_t_z_t_z_t
        if self._debug:
            print(f"batch_Z_t_z_t.shape: {batch_Z_t_z_t.shape}")
            print(f"batch_Z_t_z_t[:10]:  {batch_Z_t_z_t[:10]}")
            print()

        return batch_Z_t_z_t
    
    
    def _get_R_t_z_start_t_vec(self, batch_z_start_t, batch_t):
        """
        Return the forward rate vector R_t(z^{start}_t, :) that contains the elements 
        [R_t(z^{start}_t, :)]_{z^{end}_t}=R_t(z^{start}_t, z^{end}_t) 
        corresponding to the rates to transition from z^{start}_t to state z^{end}_t 
        at time t.

        Remark: z^{start}_t and z^{end}_t are both states at the same time that are separated
                by a 'forward in time transition' from z^{start}_t to z^{end}_t.
                The equivalent 'backward in time transition' would be from z^{end}_t to z^{start}_t.
        
        Args:
            batch_z_start_t (torch.tensor): (Batched) start state to jump from as 2D torch 
                tensor of shape (batch_size, self.x_space_dim).
            batch_t (torch.tensor): (Batched) time at which the jump occurs (for each point 
                in the batch) as 2D torch tensor of shape (batch_size, 1).
        
        Return:
            (torch.tensor): The batched forward transition rate vector R_t(z_t, :)
                as 2D torch tensor of shape (batch_size, dim_jump_state_space[z_t]).

        """
        # Determine the R_t for each of the points in the batch resulting in a 3D torch tensor
        # of shape (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)
        batch_R_t = self.R(batch_t)

        # Encode batch_z_start_t resulting in a 2D torch tensor of shape (batch_size, self.x_encoding_space_dim)
        batch_z_start_t_encoded = self.encode_x(batch_z_start_t)
        
        # R_t(z_t, :) corresponds to encoded(z_t)@R_t (where @ is matrix multiplication) because
        # encoded(z_t)=direct_sum_{d} one-hot-encoded(z^c_t) and R_t=direct_sum_{d} R^c_t (i.e. block 
        # diagonal), s.t. R_t(z_t, :)=direct_sum_{d} one-hot-encoded(z^c_t)@R^c_t = encoded(z_t)@R_t.
        batch_R_t_z_start_t_vec = utils.batched_matmul(batch_z_start_t_encoded, batch_R_t)
        if self._debug:
            print(f"batch_R_t_z_start_t_vec.shape: {batch_R_t_z_start_t_vec.shape}")
            print(f"batch_R_t_z_start_t_vec[:10]:  {batch_R_t_z_start_t_vec[:10]}")

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
                of shape (batch_size, self.x_space_dim).
            batch_t (torch.tensor): The batched time at which the jump occurs (for each point in the 
                batch) as 2D torch tensor of shape (batch_size, 1).
            train_or_eval (str): The 'train' or 'eval' mode to be used for the denoising model
                when predicting the inverse/backward rate vector.
            batch_y (None or torch.tensor): If not None, the property y as 2D torch tensor 
                of shape (1, #y-features).
                (Default: None)
        
        Return:
            (torch.tensor): The batched inverse/backward transition rate vector \hat{R}_t(z^{start}_t, :)
                as 2D torch tensor of shape (batch_size, self.x_encoding_space_dim).

        """
        # Determine the batched matrices Q_t and R_t for the batch times, that will both have 
        # shape (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)
        batch_Q_t = self.Q(batch_t)
        batch_R_t = self.R(batch_t)
        if self._debug:
            print(f"batch_Q_t.shape: {batch_Q_t.shape}")
            if batch_Q_t.is_sparse==False:
                print(f"batch_Q_t[:10]:  {batch_Q_t[:10]}")
            print(f"batch_R_t.shape: {batch_R_t.shape}")
            if batch_R_t.is_sparse==False:
                print(f"batch_R_t[:10]:  {batch_R_t[:10]}")
            print()

        # Encode batch_z_start_t that will be a 2D torch tensor of shape (batch_size, self.x_encoding_space_dim) 
        batch_z_start_t_encoded = self.encode_x(batch_z_start_t)
        
        # Slice Q_t and R_t by fixing the last (i.e. third) axis (axis=2 because axis is zero-based) 
        # indices to the entries of 'batch_z_start_t' that will lead to two 2D torch tensors of 
        # shape (batch_size, self.x_encoding_space_dim).
        # Multiply these matrices (from right) to batch_z_start_t_encoded, which works because the
        # matrices are block diagonal and the encoded batch_z_start_t contains a one-hot vector for
        # each of the blocks. Note that the block-structure reflects the different components of the 
        # (unencoded) x space.
        # I.e. obtain 2D torch tensor [Q_t]_{:, :, batch_z_start_t} and [R_t]_{:, :, batch_z_start_t}.
        batch_Q_t_vec_z_start_t = utils.batched_matmul(batch_Q_t, batch_z_start_t_encoded)
        batch_R_t_vec_z_start_t = utils.batched_matmul(batch_R_t, batch_z_start_t_encoded)        
        if self._debug:
            print(f"batch_Q_t_vec_z_start_t.shape: {batch_Q_t_vec_z_start_t.shape}")
            print(f"batch_Q_t_vec_z_start_t[:10]:  {batch_Q_t_vec_z_start_t[:10]}")
            print(f"batch_R_t_vec_z_start_t.shape: {batch_R_t_vec_z_start_t.shape}")
            print(f"batch_R_t_vec_z_start_t[:10]:  {batch_R_t_vec_z_start_t[:10]}")
            print()
        
        # Determine the batched p_theta(z|z_t) for all states z resulting in a 2D tensor of shape
        # (batch_size, self.x_encoding_space_dim).
        batch_p_theta_vec_z_start_t = self._get_p_theta_vec_z_t(batch_z_start_t_encoded, batch_t, train_or_eval, batch_y=batch_y)
        
        ######################################################################################################
        ### Determine the sum
        ### \sum_{z} q_{t|0}(z^{end}_t|z)*p^{\theta}_{0|t}(z|z^{start}_t)/q_{t|0}(z^{start}_t|z),
        ### where z runs over the discrete state space, in two steps.
        ######################################################################################################
        ### Step 1:
        # Calculate the fraction 
        # fraction_pq(z) = p^{\theta}_{0|t}(z|z^{start}_t)/q_{t|0}(z^{start}_t|z)
        # in its vectorized form, i.e. [\vec{fraction_pq}]_{z} = fraction_pq(z).
        # Remark: q_{t|0}(z^{start}_t|z) corresponds to [Q_t]_{z, z^{start}_t} in unbatched form or to
        #         batched[Q_t]_{:, z, z^{start}_t} in batched form, where the first axis is over the batch.
        #
        # Perform (batched) element-wise division of batch_p_theta_vec_z_start_t and batch_Q_t_vec_z_start_t,
        # i.e. batched[p^{\theta}_{0|t}(z|z^{start}_t)]_{j}/batched[Q_t]_{j, z, z^{start}_t} for each point 'j' in the batch
        # and each 'z' while z^{start}_t is fixed.
        # Remarks: (1) The batch points 'j' and the 'z' are both vectorized here.
        #          (2) batch_Q_t_z_start_t corresponds to [Q_t]_{:, :, batch_z_start_t} and has shape (batch_size, self.x_enc_dim).
        #          (3) batch_p_theta_vec_z_start_t has also shape (batch_size, self.x_enc_dim).
        #          (4) Use a/b = exp(log(a)-log(b)) trick for numerical stability.
        batch_fraction_pq_vec = utils.expsublog(batch_p_theta_vec_z_start_t, batch_Q_t_vec_z_start_t, eps=self._eps)
        if self._debug:
            print(f"batch_fraction_pq_vec.shape: {batch_fraction_pq_vec.shape}")
            print(f"batch_fraction_pq_vec[:10]:  {batch_fraction_pq_vec[:10]}")
            print()

        ### Step 2:
        # Determine the sum 
        # sum_qpq
        # = \sum_{z} q_{t|0}(z^{end}_t|z)*p^{\theta}_{0|t}(z|z^{start}_t)/q_{t|0}(z^{start}_t|z)
        # = \sum_{z} q_{t|0}(z^{end}_t|z)*fraction_pq(z)
        # (for each point in the batch) where 
        # fraction_pq(z) = p^{\theta}_{0|t}(z|z^{start}_t)/q_{t|0}(z^{start}_t|z) 
        # has been determined above.
        # 
        # q_{t|0}(z^{end}_t|z) corresponds to [Q_t]_{z, z^{end}_t} in unbatched and to
        # batched[Q_t]_{:, z, z^{end}_t} in batched form, where the first axis is over the batch.
        # Thus, the (batched) sum can therefore be written as 
        # batched(sum_qpq)
        # = \sum_{z} batched[q_{t|0}(z^{end}_t|z)]_{:}*batched[fraction_pq(z)]_{:}
        # = \sum_{z} batched[Q_t]_{:, z, z^{end}_t}*batched[fraction_pq]_{:, z}
        # = \sum_{z} batched[fraction_pq]_{:, z}*batched[Q_t]_{:, z, z^{end}_t}
        # = batched[fraction_pq]_{:, :}@batched[Q_t]_{:, :, z^{end}_t}
        # where @ is batch-matrix multiplication.
        # Remarks: (1) batch_fraction_pq_vec is of shape (batch_size, self.x_encoding_space_dim)
        #          (2) batch_Q_t is of shape (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)
        # TODO: MAKE THIS NUMERICALLY MORE ROBUST!!!! (Multiplication and then summation)
        batch_sum_qpq_t = utils.batched_matmul(batch_fraction_pq_vec, batch_Q_t)
        if self._debug:
            print(f"batch_sum_qpq_t.shape: {batch_sum_qpq_t.shape}")
            print(f"batch_sum_qpq_t[:10]:  {batch_sum_qpq_t[:10]}")
            print()
        ######################################################################################################

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

        # Step 2: Apply the correction for each component
        batch_hat_R_t_z_start_t_vec_c_list = list()
        for c in range(self.x_space_dim):
            # Get the vectors of R_hat and delta corresponding to the c-th encoded 
            # component which will be 2D torch tensors of shape (batch_size, self.cardinality_list[c]).
            batch_hat_R_t_z_start_t_vec_c = self._get_component_vector(batch_hat_R_t_z_start_t_vec, c)
            batch_delta_z_start_t_vec_c   = self._get_component_vector(batch_delta_z_start_t_vec, c)

            # Apply the correction for the current component
            batch_hat_R_t_z_start_t_vec_c = batch_hat_R_t_z_start_t_vec_c - batch_delta_z_start_t_vec_c*torch.sum(batch_hat_R_t_z_start_t_vec_c, dim=1).reshape(-1, 1)
            batch_hat_R_t_z_start_t_vec_c_list.append(batch_hat_R_t_z_start_t_vec_c)

        # Concatenate the corrected R_hat components to a 2D torch tensor of shape (batch_size, self.x_encoding_space_dim)
        batch_hat_R_t_z_start_t_vec = torch.cat(batch_hat_R_t_z_start_t_vec_c_list, dim=1)
        if self._debug:
            print(f"batch_delta_z_start_t_vec.shape: {batch_delta_z_start_t_vec.shape}")
            print(f"batch_delta_z_start_t_vec[:10]: {batch_delta_z_start_t_vec[:10]}")
            print(f"[corrected] batch_hat_R_t_z_start_t_vec.shape: {batch_hat_R_t_z_start_t_vec.shape}")
            print(f"[corrected] batch_hat_R_t_z_start_t_vec[:10]:  {batch_hat_R_t_z_start_t_vec[:10]}")
            print(f"Sum[:10]: {torch.sum(batch_hat_R_t_z_start_t_vec, dim=1)[:10]}")
            print()

        return batch_hat_R_t_z_start_t_vec
    
    def _get_p_theta_vec_z_t(self, batch_z_t_encoded, batch_t, train_or_eval, batch_y=None):
        """
        Determine p_theta(z|z_t) for all states z that can be reached by a transition, involving
        a change in one component (i.e. jump-transition) or no change (i.e. identity-transition 
        s.t. z=z_t) in vectorized form, 
        i.e. p_theta(:|z_t) that is a vector of shape (self.x_encoding_space_dim).

        Remark:
        Instead of working with all z that can be reached in a transition from z_t, one can also
        work with all transitions that are possible from z_t.
        In this later perspective of the problem, each transition, which include changes in one 
        component of z_t (jump-transition) or no changes (identity-transition), can be represented
        by a transition index that describes which entry in encoded(z_t) should be set to 1.
        Example: Consider a 2D discrete space with cardinalities [2, 3].
                 Allowed transitions from the state z_t=[1, 0] <=> encoded(z_t)=[0, 1, 1, 0, 0] are
                 - transition_index=0: encoded(z) = [1, 0, 1, 0, 0] (jump-transition in first discrete component)
                 - transition_index=1: encoded(z) = [0, 1, 1, 0, 0] (identity-transition in first discrete component)
                 - transition_index=2: encoded(z) = [0, 1, 1, 0, 0] (identity-transition in second discrete component)
                 - transition_index=3: encoded(z) = [0, 1, 0, 1, 0] (jump-transition in second discrete component)
                 - transition_index=4: encoded(z) = [0, 1, 0, 0, 1] (jump-transition in second discrete component)
                => Thus, there are 2 identity-transitions and 3 jump-transisions.

        Args:
            batch_z_t_encoded (torch.tensor): (Batched) encoded 'conditional' state z_t as 2D torch 
                tensor of shape (batch_size, self.x_encoding_space_dim).
            batch_t (torch.tensor): (Batched) time as 2D torch tensor of shape (batch_size, 1).
            train_or_eval (str): The 'train' or 'eval' mode to be used for the denoising model
                when determining p_theta(z|z_t).
            batch_y (torch.tensor or None): If not None; (batched) property of shape (batch_size, #y-features)
                that is used as input in case the denoising model is conditional on the property, i.e. p_theta(z|z_t, y).
                (Default: None)

        Return:
            (torch.tensor): batched[p_theta(z|z_t)] as 2D torch tensor of shape (batch_size, self.x_encoding_space_dim)
                where the first is the batch axis and the second axis describes the different encoded states z
                that can be transitioned into from z_t.
        
        """
        # Construct the model input for the current batch
        if batch_y is not None:
            batch_model_input = {'x': batch_z_t_encoded, 'y': batch_y}
        else:
            batch_model_input = {'x': batch_z_t_encoded}

        # Set the denoising model into 'train mode' or 'eval mode'
        if train_or_eval=='train':
            self.model_dict['denoising'].train()
        elif train_or_eval=='eval':
            self.model_dict['denoising'].eval()
        else:
            err_msg = f"The input 'train_or_eval' must be either 'train' or 'eval'."
            raise ValueError(err_msg)
        
        # Use the prediction model to predict the 'energies' E^theta^{theta}_{0|t}(z|z_t)=log( p^{theta}_{0|t}(z|z_t) )
        # for fixed z_t and all z in the state space, resulting in a tensor of shape (batch_size, self.x_encoding_space_dim).
        batch_E_theta_vec_z_t = self.model_dict['denoising'](batch_model_input, batch_t)

        # Loop over the components 'c' of x and apply a softmax over the energies in each segment (i.e. subspace) within
        # the encoded x-space corresponding to the component, to obtain normalized probabilities for each component:
        # p^{theta}_{0|t}(z^{c}|z_t)
        # = Softmax(E^theta^{theta}_{0|t}(z^{c}|z_t))
        # = exp( E^theta^{theta}_{0|t}(z^{c}|z_t) )/[ sum_{z'^c} exp( E^theta^{theta}_{0|t}(z'^c|z_t) ) ]
        # for each component 'c' (where the sum is over all possible states z'^c in component 'c')
        # resulting in the normalized probabilitity vectors of shape (batch_size, cardinality_c).
        # The resulting normalized probability vectors are then concatenated over the feature (i.e. second) axis to a 
        # 2D torch tensor of shape (batch_size, self.x_encoding_space_dim) that itself is not normalized!
        batch_p_theta_vec_c_z_t_list = list()
        for c in range(self.x_space_dim):
            # Get the energy vector corresponding to the c-th encoded component which 
            # will be a 2D torch tensor of shape (batch_size, self.cardinality_list[d]).
            batch_E_theta_vec_c_z_t = self._get_component_vector(batch_E_theta_vec_z_t, c)

            # Apply the softmax function to the probabilities of the current component
            batch_p_theta_vec_c_z_t = self.softmax_fn(batch_E_theta_vec_c_z_t)
            batch_p_theta_vec_c_z_t_list.append(batch_p_theta_vec_c_z_t)

        # Concatenate the component specific probability vectors to a 2D torch tensor of shape
        # (batch_size, self.x_encoding_space_dim)
        batch_p_theta_vec_z_t = torch.cat(batch_p_theta_vec_c_z_t_list, dim=1)
        if self._debug:
            print(f"batch_E_theta_vec_z_t.shape: {batch_E_theta_vec_z_t.shape}")
            print(f"batch_E_theta_vec_z_t[:10]:  {batch_E_theta_vec_z_t[:10]}")
            print()
            print(f"batch_p_theta_vec_z_t.shape: {batch_p_theta_vec_z_t.shape}")
            print(f"batch_p_theta_vec_z_t[:10]:  {batch_p_theta_vec_z_t[:10]}")
            print()

        return batch_p_theta_vec_z_t


    def forward_sample_z_t(self, batch_x, t):
        """
        Sample z_t in forward diffusion (i.e. noised state at time t).
        
        Args:
            batch_x (torch.tensor): Original (i.e. unnoised) x-data as 2D 
                tensor of shape (batch_size, self.x_space_dim).
            t (torch.tensor, float or int): Time to sample z_t at.

        Return:
            (torch.tensor): Sampled z_t for time 't' and original x-data 'x'
                as 2D tensor of shape (batch_size, self.x_space_dim).

        """
        # Get batch size and use it to parse the time (i.e. ensure that the time is batched)
        batch_size = batch_x.shape[0]
        batch_t    = self._parse_time(t, batch_size)
        #print(f"t.shape: {t.shape}")

        # Determine the batched Q_t for the time of each point in the batch resulting in a 3D
        # torch tensor of shape (batch_size, self.x_encoding_space_dim, self.x_encoding_space_dim)
        batch_Q_t = self.Q(batch_t)

        # Encode batched x resulting in a 2D torch tensor of shape (batch_size, self.x_encoding_space_dim)
        batch_x_encoded = self.encode_x(batch_x)

        # Determine the probability vector of the categorical distribution at time t
        # for each component d of the state.
        # As Q_t will be block diagonal (in the second and third axis), we can obtain 
        # these probabilities by multiplying x_encoded from left to Q_t, i.e. for each
        # component we have
        # prob_vec^c_t = one-hot-encode(x^c)*Q^c_t
        # where the each entry in prob_vec^c_t corresponds to the probability of the
        # category encoded by the corresponding entry in one-hot-encode(x^c).
        # As 
        # encode(x) = direct-sum_{d}one-hot-encode(x^c) 
        # and 
        # Q_t = direct-sum_{d}Q^c_t
        # the following holds
        # prob_vec_t = direct_sum_{d} prob_vec^{c}_t
        #            = direct_sum_{d} one-hot-encode(x^c)*Q^c_t
        #            = encode(x)*Q_t
        # Remarks: (1) Use batched matrix multiplication (bmm) in the form
        #              sum_{i=1}^{self.x_encoding_space_dim} [x_encoded]_(b,i)[Q_t]_(b,i,j)
        #              where b is the index of a point in the batch.
        #          (2) The result batch_prob_vec_t is a 2D torch tensor of shape 
        #              (batch_size, self.x_encoding_space_dim)
        batch_prob_vec_t = utils.batched_matmul(batch_x_encoded, batch_Q_t)
        if self._debug:
            print(f"batch_prob_vec_t.shape: {batch_prob_vec_t.shape}")
            print(f"batch_prob_vec_t[:10]:  {batch_prob_vec_t[:10]}")
            #raise ValueError("AM HERE NOW")
            print()

        # Loop over the different components 'c' of the x-space and sampled z^c_t for each component
        # thereby obtaining the sampled components z^c_t of shape (batch_size, 1). Concatenate these
        # components along the feature (i.e. second) axis to obtain a 2D torch torch tensor of shape
        # (batch_size, self.x_space_dim) where each row corresponds to the sampled z_t for the
        # respective point in the batch.
        batch_z_t = torch.cat([self._sample_z_c_t(batch_prob_vec_t, c) for c in range(self.x_space_dim)], dim=1)
        if self._debug:
            print(f"batch_z_t.shape: {batch_z_t.shape}")
            print(f"batch_z_t[:10]:  {batch_z_t[:10]}")

        return batch_z_t
    
    def _sample_z_c_t(self, prob_vec_t, c):
        """
        Sample z^c_t, the c-th component of z_t, based on the probability vector 
        within the subspace of the encoded space corresponding to of the c-th 
        discrete component.
        
        Args:
            prob_vec_t (torch.tensor): The probability vector in the encoded x-space as 2D torch
                tensor of shape (batch_size, self.x_encoding_space_dim) where segments corresponding
                to a specific component in the x-space are normalized.
                => I.e. prob_vec_t is not normalized over the feature (i.e. second axis).
            c (int): The component of z_t to be sampled.

        Return:
            (torch.tensor): Sampled z^c_t as 2D tensor of shape (batch_size, 1).

        """
        # Get the probability vector corresponding to the c-th encoded component which 
        # will be a 2D torch tensor of shape (batch_size, self.cardinality_list[d])
        # and which will be properly normalized.
        prob_vec_c_t = self._get_component_vector(prob_vec_t, c)
        if self._debug:
            print(f"prob_vec_c_t.shape: {prob_vec_c_t.shape}")
            print(f"prob_vec_c_t[:10]:  {prob_vec_c_t[:10]}")
            print()
        
        # Sample z^c_t from a categorical distribution with this probability vector prob_vec^c_t
        q_z_c_t = torch.distributions.categorical.Categorical(prob_vec_c_t) 
        z_c_t   = q_z_c_t.sample()
        if self._debug:
            print(f"z_c_t.shape: {z_c_t.shape}")
            print(f"z_c_t[:10] : {z_c_t[:10]}")

        # Return z^c_t
        # Remark: To be able to concatenate z^c_t entry for each component c collected in a list 
        #         (outside this method), they must be from a 1D torch tensor of shape (batch_size,) 
        #         to a 2D torch tensor of shape (batch_size, 1).
        return z_c_t.reshape(-1, 1)

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
                (batch_size, self.x_space_dim) generated in backward diffusion.
        
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
                (1, self.x_space_dim) generated in backward diffusion.
        
        """
        # Initialize the current (i.e. last) time 't_last' as 2D torch tensor of shape (1, 1) containing the value 1
        # Remark: We always use times as 2D tensors of shape (batch_size, 1) but for the purpose here the batch_size is 1.
        t_last = torch.tensor(1).reshape(1, 1)

        # Sample z_(t=1)=z_1 and set it as the initial current (i.e. last) state 'z_t_last'
        # as 2D torch tensor of shape (1, self.x_space_dim)
        z_t_last = self._sample_z_1(batch_size=1).reshape(1, -1)
        if self._debug:
            print(f"z_1.shape: {z_t_last.shape}")
            print(f"z_1[:10]:  {z_t_last[:10]}")

        # Loop over the time steps
        for _ in range(max_num_time_steps):
            # Sample the next time, i.e. by holding the state z_t_last for the
            # holding time equals to t_last-t_next, as 0D torch tensor (i.e. scalar).
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
                # Return z_t_last (that is the final state at t=0 here) and the value of t_next
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
                2D torch tensor of shape (1, self.x_space_dim). 
            num_integration_points (int): Number of integration points used to
                approximate the integral int_{t_last}^{t_next}\hat{Z}_s(z_t_last)ds
                when sampling the next time t_next.
                (Default: 100)

        Return:
            (torch.tensor): Sampled next time as 2D torch tensor of shape (1, 1).
        
        """
        
        # Expand z_t_last, which is a 0D torch tensor to the number of integration steps
        # so that each integration step has the same z_t, resulting in a 2D torch tensor
        # of shape (#integration_points, self.x_space_dim)
        expanded_z_t_last = z_t_last.expand(num_integration_points, -1)
        if self._debug:
            print(f"expanded_z_t_last.shape: {expanded_z_t_last.shape}")
            print(f"expanded_z_t_last[:10]: {expanded_z_t_last[:10]}")

        # Determine the time integration points {t_j}_{j=0}^{N-1} where t_0=t_last and t_{N-1}=0 
        # as 2D torch tensor of shape (#integration_points, 1)
        # Remark: t_last is a 2D tensor of shape (1, 1), use its value [].item()] as float for torch.linspace.
        t_intpnts = torch.linspace(t_last.item(), 0, num_integration_points).reshape(-1, 1)

        # Also determine the absolute values of their time-differences, i.e. t_diff_j = |t_{j+1}-t_{j}|
        # as 1D torch tensor of shape (#integration_points-1,)
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
                tensor of shape (1, self.x_space_dim).
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


        # Determine the probability vector to jump from z_t_last to any other state
        # that differs from z_t_last in only a single component at time t_next. 
        # This probability vector will be a 2D torch tensor of shape (batch_size, self.x_enc_dim).
        # Remark: The probabilities are over the encoded x-space, i.e. they specify what 
        #         the probability is to change to a new specific entry in encoded(z_t_last).
        prob_vec_jump_t_next_z_t_last = self._get_jump_prob_vec(z_t_last, hat_R_t_next_z_t_vec)
        if self._debug:
            print(f"prob_vec_jump_t_next_z_t_last.shape: {prob_vec_jump_t_next_z_t_last.shape}")
            print(f"prob_vec_jump_t_next_z_t_last:       {prob_vec_jump_t_next_z_t_last}")

        ###################################################################################################
        ### Property guidance - start
        ###################################################################################################
        # In case that the property model exists (i.e. is not None), determine 
        # the gradient of it w.r.t. to z_t_last
        if self.model_dict['property'] is not None:   
            if y is None:
                err_msg = f"Can only use property guidance in case that 'y' is passed to 'generate' method."
                raise ValueError(err_msg)
            
            # Create a torch tensor holding the transition indices specifying the transitions to any allowed states
            # that can be reached from z_t_last by either no change (identity-transition), i.e. z=z_t_last, or a change 
            # in only one discrete component of z_t_last (jump-transition).
            # Teh transitions include self.x_space_dim identity-transitions and (self.x_encoding_space_dim-self.x_space_dim) 
            # jump-transitions so that #transitions=self.x_encoding_space_dim and thus 'batch_transit_index_t_next'
            # is a 2D torch tensor of shape (self.x_encoding_space_dim, 1).
            transit_indices_t_next     = torch.arange(self.x_encoding_space_dim).reshape(-1, 1)
            z_transit_to_from_z_t_last = self._transition_to_new_state(z_t_last, transit_indices_t_next)
            if self._debug:
                print(f"z_transit_to_from_z_t_last.shape: {z_transit_to_from_z_t_last.shape}")
                print(f"z_transit_to_from_z_t_last[:10]:  {z_transit_to_from_z_t_last[:10]}")
                print(f"y.shape: {y.shape}") 
                print(f"y: {y}")

            # Expand the next time to a torch tensor of shape (#transitions, 1) where #transitions=self.x_encoding_space_dim
            expanded_t_next = t_next.expand(z_transit_to_from_z_t_last.shape[0], -1)
            if self._debug:
                print(f"expanded_t_next.shape: {expanded_t_next.shape}")
                print(f"expanded_t_next[:10]:  {expanded_t_next[:10]}")

            # Expand the y values
            expanded_y = y.expand(z_transit_to_from_z_t_last.shape[0], -1)
            if self._debug:
                print(f"expanded_y.shape: {expanded_y.shape}")
                print(f"expanded_y[:10]:  {expanded_y[:10]}")

            # Determine the log-probability (i.e. log-likelihood) of the property 
            # model for each of the states that can be transitioned to from z_t_last. 
            # The result 'log_prob' will be a 1D torch tensor of shape (#transitions,)
            # where #transitions=self.x_encoding_space_dim
            #log_prob = self.model_dict['property'].log_prob(z_transit_to_from_z_t_last_encoded, expanded_t_next, expanded_y)
            log_prob = self._get_property_model_log_prob(z_transit_to_from_z_t_last, expanded_t_next, expanded_y)
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
                delta_z_t_next_z_t_last = self.encode_x(z_t_last)
                prob_vec_jump_t_next_z_t_last  = (1-delta_z_t_next_z_t_last)*prob_vec_jump_t_next_z_t_last
            
                # Fourth, normalize the probabilities
                prob_vec_jump_t_next_z_t_last = prob_vec_jump_t_next_z_t_last/torch.sum(prob_vec_jump_t_next_z_t_last)
            
            # Display results for the user
            if self._debug:
                print(f"[guided]prob_vec_jump_t_next_z_t_last.shape: {prob_vec_jump_t_next_z_t_last.shape}")
                print(f"[guided]prob_vec_jump_t_next_z_t_last[:10]:  {prob_vec_jump_t_next_z_t_last[:10]}")

        ###################################################################################################
        ### Property guidance - end
        ###################################################################################################
        # Sample the 'jump index' from a categorical distribution with the probability vector 
        # p^{jump}_t_{next}[z_t] that corresponds to p^{jump}_t_{next}(jump_index|z_t) where the jump_index
        # is an integer in [0, self.x_encoding_space_dim-1] and specifies the index in the encoded
        # x space that will be the new state of the corresponding component.
        q_jump_index_z_t_next   = torch.distributions.categorical.Categorical(prob_vec_jump_t_next_z_t_last) 
        batch_jump_index_t_next = q_jump_index_z_t_next.sample()

        # Jump-transition to the new state at t_next
        z_jump_t_next = self._transition_to_new_state(z_t_last, batch_jump_index_t_next)
        
        if self._debug:
            print(f"z_t_last.shape: {z_t_last.shape}")
            print(f"z_t_last[:10]:  {z_t_last[:10]}")
            print(f"z_jump_t_next.shape: {z_jump_t_next.shape}")
            print(f"z_jump_t_next[:10]:  {z_jump_t_next[:10]}")
            print()

        return z_jump_t_next
    
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

        # Loop over the different components 'c' of the x-space and sampled z^c_t for each component
        # thereby obtaining the sampled components z^c_t of shape (batch_size, 1). Concatenate these
        # components along the feature (i.e. second) axis to obtain a 2D torch torch tensor of shape
        # (batch_size, self.x_space_dim) where each row corresponds to the sampled z_t for the
        # respective point in the batch.
        batch_z_1 = torch.cat([self._sample_z_c_t(batch_prob_vec_1, c) for c in range(self.x_space_dim)], dim=1)
        if self._debug:
            print(f"batch_z_1.shape: {batch_z_1.shape}")
            print(f"batch_z_1[:10]:  {batch_z_1[:10]}")

        return batch_z_1
    
    def _get_property_model_log_prob(self, batch_z_t, batch_t, batch_y):
        """
        Return the log probability of the property log model.
        
        Args:
            batch_z_t (torch.tensor): (Batched) state z_t as 2D torch tensor
                of shape (batch_size, self.x_space_dim).
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

    
