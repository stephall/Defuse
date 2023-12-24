# Import public modules
import torch
import numpy as np
from importlib import reload

# Import custom modules
from . import base_diffusion_manager

# Reload custom modules
reload(base_diffusion_manager)

class ContinuousDiffusionManager(base_diffusion_manager.BaseDiffusionManager):
    """
    Define a manager to handle continuous time diffusion in continuous spaces.
    """
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

        # Sample latent state of each datapoint in the batch for their
        # corresponding time t (in forward diffusion process)
        # Remark: 'self.forward_sample_z_t' returns the state and the noise 'epsilon' if 'return_epsilon=True'
        batch_z_t, batch_epsilon = self.forward_sample_z_t(batch_x, batch_t, return_epsilon=True)

        # Construct the model input for the current batch
        if 'y' in batch_data:
            batch_model_input = {'x': batch_z_t, 'y': batch_data['y']}
        else:
            batch_model_input = {'x': batch_z_t}

        # Set the denoising model into 'train mode'
        self.model_dict['denoising'].train()
        
        # Predict the noise of the batch epsilons
        batch_pred_epsilon = self.model_dict['denoising'](batch_model_input, batch_t)

        # Determine the squared difference between the 'ground truth' and
        # predicted epsilon tensors
        # Remark: Sum over the feature axis (i.e. the second axis) to obtain
        #         the difference.
        batch_squared_diff = torch.sum((batch_epsilon-batch_pred_epsilon)**2, dim=1)

        # Get the derivative of gamma for the times of the batch
        #batch_deriv_gamma_t = self.deriv_gamma(batch_t)
        #print(f"squared_diff.shape: {batch_squared_diff.shape}")
        #print(f"deriv_gamma_t.shape: {batch_deriv_gamma_t.shape}")

        # Get the time-dependent weight for the loss function
        batch_time_weight_t = self.time_weight(batch_t)

        # Calculate the loss for each datapoint in the batch and then
        # sum over all of these to obtain the total loss of the batch
        loss = 0.5*torch.sum(batch_time_weight_t*batch_squared_diff)
        #loss = 0.5*torch.sum(batch_squared_diff)
        #print(f"loss.shape: {loss.shape}")
        #print(f"loss: {loss}")

        return loss

    def forward_sample_z_t(self, x, t, return_epsilon=False):
        """
        Sample z_t in forward diffusion (i.e. noised state at time t).
        
        Args:
            x (torch.tensor): Original (i.e. unnoised) x-data as 2D tensor
                of shape (batch_size, #x-features).
            t (torch.tensor, float or int): Time to sample z_t at.
            return_epsilon (bool): Boolean flag that specifies if epsilon
                (i.e. the noise used to sample z_t) should be returned or not.
                (Default: Dalse)

        Return:
            Cases:
            return_epsilon=True:
                (torch.tensor): Sampled z_t for time 't' and original x-data 'x'
                    as 2D tensor of shape (batch_size, #x-features).
            return_epsilon=False:
                (torch.tensor, torch.tensor): Sampled z_t for time 't' and 
                    original x-data 'x' as 2D tensor of shape (batch_size, 
                    #x-features) as first return variable and the epsilon
                    (i.e. the noise used to sample z_t) also as 2D tensor of
                    shape (batch_size, #x-features) as second return variable.

        """
        # Get batch size
        batch_size = x.shape[0]

        # Parse the time t
        t = self._parse_time(t, batch_size)

        # Get kappa^2 and sigma^2
        kappa2_t = self.kappa2(t)
        sigma2_t = self.sigma2(t)

        # Sample epsilon as random tensor of shape (batch_size, #x-features)
        # (i.e. the same shape as x) whose entries are each i.i.d. drawn 
        # from normal distribution N(0, 1)
        epsilon = torch.randn_like(x)

        if self._debug:
            print('Shapes:')
            print(f"t.shape: {t.shape}")
            print(f"kappa2_t.shape: {kappa2_t.shape}")
            print(f"sigma2_t.shape: {sigma2_t.shape}")
            print(f"x.shape: {x.shape}")
            print(f"epsilon.shape: {epsilon.shape}")
            print('Values:')
            print(f"t: {t[:10]}")
            print(f"kappa2_t: {kappa2_t[:10]}")
            print(f"sigma2_t: {sigma2_t[:10]}")
            print()

        # Use the epsilon samples to obtain samples from the
        # noised distribution at time t
        z_t = torch.sqrt(kappa2_t)*x + torch.sqrt(sigma2_t) * epsilon

        # Return values depending on the boolean flag 'return_epsilon'
        if return_epsilon:
            # If 'return_epsilon' is True, also return the epsilon values
            # used to 'sample' the z_t
            return z_t, epsilon
        else:
            # Otherwise, do only return z_t and not return these epsilon values
            return z_t
        
    def generate(self, batch_size=100, num_time_steps=100, random_seed=24, y=None):
        """
        Generate 'novel' points by sampling from p(z_1) and then using ancestral 
        sampling (backwards in time) to obtain 'novel' \hat{z}_0=\hat{x}.

        Args:
            batch_size (int): Number of 'novel' points '\hat{x}' to generate.
                (Default: 100)
            num_time_steps (int): Number of ancestral sampling (time) steps to 
                use for backward propagation through time.
                (Default: 100)
            random_seed (int): Random seed to be used for all the sampling in
                the generation process.
                (Default: 24)
            y (int or None): Conditional class to guide to.
        
        Return:
            (torch.tensor): 'Novel' points as 2D torch tensor \hat{x} of shape
                (batch_size, #x-features) generated in backward diffusion.
        
        """
        # Parse y if it is not None
        if y is not None:
            y = y*torch.ones(batch_size, dtype=torch.int).reshape(-1, 1)

        # Set a random seed
        self.set_random_seed(random_seed)

        # Generate a grid of decreasing t values in [0, 1]
        # Remark: Use this uniform grid following the article 
        #         'Classifier Free Diffusion Guidance'
        #          (https://arxiv.org/pdf/2207.12598.pdf) 
        #          by J. Ho & T. Salimans.
        time_list = list(np.linspace(1, 0, num_time_steps))
        if self._debug:
            print(f"time_list length: {len(time_list)}")
            print(f"time_list: {time_list}")

        # Sample z_(t=1)=z_1 and set it as the initial z_t
        z_t = self._sample_z_1(batch_size)
        if self._debug:
            print(f"z_t.shape: {z_t.shape}")

        # Loop over the times (denote the time step index as 'n' in documentation)
        # Remark: We are going backwards in time so that t_lat_float>t_next_float.
        for step, (t_last_float, t_next_float) in enumerate(zip(time_list[:-1], time_list[1:])):
            if self._debug:
                print()
                print(f"t_last = t_(n)   = {t_last_float}")
                print(f"t_next = t_(n+1) = {t_next_float}")

            # Determine the batch size from z_t
            batch_size = z_t.shape[0]

            # Cast t_last and t_next to 2D torch tensors of shape (batch_size, 1)
            # whoses entries are identical
            t_last = self._parse_time(t_last_float, batch_size)
            t_next = self._parse_time(t_next_float, batch_size)

            # Sample (backward in time) the next z_t (i.e. z_{t_{n+1}}) based on 
            # last (i.e. current) z_t (i.e. z_{t_{n}})
            z_t = self._backward_sample_z_t_next(z_t, t_last, t_next, y)

        # Return the final z_t
        return z_t

    def _backward_sample_z_t_next(self, z_t_last, t_last, t_next, y=None):
        """
        Sample next z_{t_{n+1}} backward in time from last (i.e. current) 
        z_{t_{n}}, which corresponds to input z_t_last, and where 
        t_last=t_{n}>t_{n+1}=t_next (because we are going backward in time).
        
        Remark:
        Use backward parametrization as specified in 'Variational Diffusion Models'
        (https://arxiv.org/pdf/2107.00630.pdf) by D. Kingma, T. Salimans, 
        B. Poole, and J. Ho where 'kappa^2(t)' corresponds to 'alpha^2(t)' in 
        the article.

        Args:
            z_t_last (torch.tensor): The last (i.e. current) z_t as 2D torch 
                tensor of shape (batch_size, #x-features).
            t_last (torch.tensor): The last (i.e. current) times over the points
                in the batch as 2D torch tensor of shape (batch_size, 1).
            t_next (torch.tensor): The next times over the points in the batch 
                as 2D torch tensor of shape (batch_size, 1).
            y (None or torch.tensor): If not None, the property y for each point 
                in the batch as 2D torch tensor of shape 
                (#batch_size, #y-features).
                (Default: None)

        Return:
            (torch.tensor): Sampled z_t_next=z_{t_{n+1}} as torch tensor of the
                same shape as the last (i.e. current) z_t_last=z_{t_{n}}.

        """
        # Determine kappa^2(t_last), kappa^2(t_next), gamma(t_last), 
        # and gamma(t_next) where t_last=t_{n} and t_next=t_{n+1}.
        kappa2_t_last = self.kappa2(t_last)
        kappa2_t_next = self.kappa2(t_next)
        gamma_t_last  = self.gamma(t_last)
        gamma_t_next  = self.gamma(t_next)

        if self._debug:
            print(f"kappa2_t_last[:10]: {kappa2_t_last[:10]}")
            print(f"kappa2_t_next[:10]: {kappa2_t_next[:10]}")

        # Construct the input for the noiding modelat t_last (containing y if passed).
        if y is None:
            model_input_t_last = {'x': z_t_last}
        else:
            model_input_t_last = {'x': z_t_last, 'y': y}

        # Don't calculate any gradients when predicting using the 'denoising' model
        with torch.no_grad():
            # Set the denoising model into 'eval mode'
            self.model_dict['denoising'].eval()

            # Predict epsilon at the last (i.e. current) time t_last=t_{n}
            pred_epsilon_t_last = self.model_dict['denoising'](model_input_t_last, t_last)

        ###################################################################################################
        ### Property guidance - start
        ###################################################################################################
        # In case that the property model exists (i.e. is not None), determine 
        # the gradient of it w.r.t. to z_t_last
        if self.model_dict['property'] is not None:
            if y is None:
                err_msg = f"Can only use property guidance in case that 'y' is passed to 'generate' method."
                raise ValueError("AM HERE")

            # Avoid any training and gradient building, so clone (i.e. deepcopy)
            # z_t_last (that should not require gradients) and make its clone/copy 
            # _z_t_last require gradients.
            # This clone/copy with required gradient is used as input to obtain
            # the gradient of the log-probability of the prediction model w.r.t.
            # to z_t_last below.
            # Remark: Calling 'requires_grad_()' on '_z_t_last' will not set
            #         'requires_grad' to True in the original 'z_t_last'.
            _z_t_last = torch.clone(z_t_last)
            _z_t_last.requires_grad_() # Inplace set that _z_t_last requires gradients

            # Set the model to 'eval-mode'
            self.model_dict['property'].eval()

            # Determine the gradient of the model w.r.t. the input z_t_last
            # following the source:
            # https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch
            # Remark: Although x and thus the gradients would be batched, we are not using 
            #         'is_grads_batched' here because we don't need it. 
            log_prob_t_last = self.model_dict['property'].log_prob(_z_t_last, t_last, y)
            grad_log_prob_t_last = torch.autograd.grad(outputs=log_prob_t_last, 
                                                       inputs=_z_t_last, 
                                                       grad_outputs=torch.ones_like(log_prob_t_last), 
                                                       create_graph=True)[0]

            # As described above, we do not want to require gradients here if 
            # not needed for the gradient calculation above. Thus, use 'detach'
            # on 'grad_log_prob_t_last' to detach it from the computation grad 
            # to remove its 'grad_fn'
            grad_log_prob_t_last = grad_log_prob_t_last.detach()


            #print(f"_z_t_last: {_z_t_last}")
            #print(f"z_t_last: {z_t_last}")
            #print(f"grad_log_prob_t_last.shape: {grad_log_prob_t_last.shape}")
            #print(f"grad_log_prob_t_last[:10]: {grad_log_prob_t_last[:10]}")

            # Add '-sigma_t_last*gradient(log_prob_t_lasst)' to the predicted epsilon
            # for property guidance.
            # Remark: sigma_t_last = sqrt( 1-kappa2_t_last )
            pred_epsilon_t_last = pred_epsilon_t_last-torch.sqrt(1-kappa2_t_last)*grad_log_prob_t_last
            
            #print(f"pred_epsilon_t_last[:10]: {pred_epsilon_t_last[:10]}")

        ###################################################################################################
        ### Property guidance - end
        ###################################################################################################

        # Determine several factors
        kappa_fac = torch.sqrt(kappa2_t_next/kappa2_t_last)
        expm1_fac = -torch.expm1(gamma_t_next-gamma_t_last)
        delta_z_t = -torch.sqrt(1-kappa2_t_last)*expm1_fac*pred_epsilon_t_last

        if self._debug:
            print(f"kappa_fac[:10]: {kappa_fac[:10]}")
            print(f"expm1_fac[:10]: {expm1_fac[:10]}")
            print(f"delta_z_t[:10]: {delta_z_t[:10]}")

        # Calculate the mean and sigma of the distribution of z_(t-1)
        mu_t_next    = kappa_fac * (z_t_last + delta_z_t)
        sigma_t_next = torch.sqrt( (1-kappa2_t_next)*expm1_fac )

        # Sample xi from a normal distribution N(0, 1) as tensor of the same
        # shape as z_t_last where all entries are drawn i.i.d.
        xi = torch.randn_like(z_t_last)

        if self._debug:
            print(f"xi.shape: {xi.shape}")
            print(f"xi: {xi}")
            print(f"mu_t_next.shape: {mu_t_next.shape}")
            print(f"mu_t_next[:10]: {mu_t_next[:10]}")
            print(f"sigma_t_next.shape: {sigma_t_next.shape}")
            print(f"sigma_t_next[:10]: {sigma_t_next[:10]}")

        # The sample of z_next=z_{t_{n+1}} is determined based on the sampled xi
        # and mu(t_next) and sigma(t_next)
        z_t_next = mu_t_next + sigma_t_next * xi
        if self._debug:
            print(f"z_t_next.shape: {z_t_next.shape}")
            print(f"z_t_next[:10]: {z_t_next[:10]}")

        # Return the sampled z_t_next
        return z_t_next
    
    def _sample_z_1(self, batch_size):
        """
        I.i.d. drawn z_1 for each point in a batch. 

        Args:
            batch_size (int): Batch size.

        Return:
            (torch.tensor): 2D torch tensor corresponding to a z_1 sample that
                will have shape (batch_size, self.x_num_features).
        
        """
        # Define the loc and std parameters of a standardized normal distribution
        # in the shape of (batch_size, #x-features) where the #x-features have
        # been determined during training from the 'x-data'.
        loc = torch.zeros(batch_size, self.x_num_features)
        std = torch.ones(batch_size, self.x_num_features)

        # Sample z_1 as i.i.d. samples drawn from a standardized normal 
        # distribution and cast into a tensor of shape (batch_size, #x-features) 
        # corresponding to z_1 and return it
        return torch.normal(loc, std)
    
    def _get_property_model_log_prob(self, batch_z_t, batch_t, batch_y):
        """
        Return the log probability of the property log model.
        
        Args:
            batch_z_t (torch.tensor): (Batched) state z_t as 2D torch tensor
                of shape (batch_size, state_space_dimension).
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
        
        # Determine the log-probability (i.e. log-likelihood) of the property 
        # model for the batch data for each point in the batch, i.e. log_prob 
        # is a 1D torch tensor of shape (batch_size,)
        return self.model_dict['property'].log_prob(batch_z_t, batch_t, batch_y)