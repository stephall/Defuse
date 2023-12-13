# Import public modules
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

# Import custom modules
from src import encodings
from src import utils

# Reload custom modules
reload(encodings)
reload(utils)

class BaseDiffusionManager(object):
    """
    Define the base class for any type of 'diffusion'.
    """
    # Define the default SNR min and max values
    _default_SNR_min = 1e-4
    _default_SNR_max = 1/_default_SNR_min

    # Define the default gamma scale factor
    # Remark: Use 2 as default value as defined in the article 
    #         'Classifier Free Diffusion Guidance'
    #         (https://arxiv.org/pdf/2207.12598.pdf) by J. Ho & 
    #         T. Salimans where 'gamma(t)' corresponds to '-lambda(u)'.
    _default_gamma_scale = 2

    # Define the default time weighting strategy
    _default_time_weighting_strategy = 'uniform'

    def __init__(self, denoising_model, denoising_optimizer, property_model=None, property_optimizer=None, SNR_min=None, SNR_max=None, gamma_scale=None, time_weighting_strategy=None, debug=False):
        # Assign inputs to class attributes
        self.model_dict = {
            'denoising': denoising_model,
            'property':  property_model,
        }
        self.optim_dict = {
            'denoising': denoising_optimizer,
            'property':  property_optimizer,
        }
        self._debug = debug

        #########################################################################################################
        ### Parse SNR_min, SNR_max, and gamma_scale, and assign them to corresponding class attributes.
        #########################################################################################################
        # 1) Initialize the class attributes using their default values.
        #    Remark: Both methods 'set_SNR_min_max' and 'set_gamma_scale' will call 'set_gamma_params'
        #            internally, that will set self._gamma_param_a and self._gamma_param_b depending on
        #            the current values of self.SNR_min, self.SNR_max, and self.gamma_scale.
        #            Thus, using the default values ensures that this method can be called.
        #            I.e. the first call of 'set_gamma_params' within 'set_SNR_min_max' will use the
        #            'passed values' of 'SNR_min' and 'SNR_max' but the default value of 'gamma_scale',
        #            but the second call of 'set_gamma_params' within 'set_gamma_scale' will use the 'passed value'
        #            of 'gamma_scale'. 
        self.SNR_min     = self._default_SNR_min
        self.SNR_max     = self._default_SNR_max
        self.gamma_scale = self._default_gamma_scale

        # 2) If any of them is not passed, also use their default values to update the 
        #    'passed values'
        if SNR_min is None:
            SNR_min = self._default_SNR_min
        if SNR_max is None:
            SNR_max = self._default_SNR_max
        if gamma_scale is None:
            gamma_scale = self._default_gamma_scale

        # 3) Set SNR_min and SNR_passed using the 'passed values' using 'set_SNR_min_max' 
        #    (that also checks the validity of 'SNR_min' and 'SNR_min' before setting them 
        #    to the corresponding class attributes).
        self.set_SNR_min_max(SNR_min, SNR_max)

        # 4) Set gamma_scale using 'set_gamma_scale' (that also checks the validity of 'gamma_scale' 
        #    before setting them to the corresponding class attributes).
        self.set_gamma_scale(gamma_scale)
        
        #########################################################################################################
        #########################################################################################################

        # Parse the time weighting strategy, using the default strategy if it is not passed (i.e. None)
        if time_weighting_strategy is None:
            time_weighting_strategy = self._default_time_weighting_strategy
            
        self.time_weighting_strategy = time_weighting_strategy

        # Define the sigmoid function
        self.sigmoid_fn = torch.nn.Sigmoid()

        # Initialize certain class attributes to None that will be set during
        # usage of an instance
        self.x_num_features = None

    def set_SNR_min_max(self, SNR_min, SNR_max):
        """
        Set the SNR_min and SNR_max values to their corresponding class 
        attributes self.SNR_min and self.SNR_max.
        
        Args:
            SNR_min (float or int): Minimal signal-to-noise ration (SNR) that 
                must be a positive number and smaller than SNR_max (below).
            SNR_min (float or int): Maximal signal-to-noise ration (SNR) that 
                must be a positive number and larger than SNR_min (above). 
            
        Return:
            None
        
        """
        # 1) Parse both of SNR_min and SNR_max
        SNR_min = self._parse_SNR_min_or_max('SNR_min', SNR_min)
        SNR_max = self._parse_SNR_min_or_max('SNR_max', SNR_max)
    
        # 2) Check that SNR_max is larger than SNR_min
        if SNR_max<=SNR_min:
            err_msg = f"'SNR_max' must be larger than 'SNR_min', but 'SNR_min={SNR_min}' and 'SNR_max={SNR_max}' were passed."
            raise ValueError(err_msg)

        # 3) Assign them to class attributes of the same name
        self.SNR_min = SNR_min
        self.SNR_max = SNR_max

        # 3) Set the gamma parameter (that depend on self.SNR_min and self.SNR_max)
        self.set_gamma_params()

    def _parse_SNR_min_or_max(self, which_SNR, SNR_value):
        """
        Parse SNR_min or SNR_max.
        
        Args:
            which_SNR (str): Either 'SNR_min' or 'SNR_max'.
            SNR_value (int or float): The value of either 'SNR_min' or 'SNR_max'.

        Return:
            (torch.tensor): SNR_value as float scalar tensor.
        
        """
        # Ensure that SNR_value is a number
        if not isinstance(SNR_value, (int, float)):
            err_msg = f"'{which_SNR}' must be positive numbers (int or float), got type '{type(SNR_value)}' instead."
            raise TypeError(err_msg)

        # Check that SNR_value is larger than zero
        if SNR_value<=0:
            err_msg = f"'{which_SNR}' must be positive numbers (int or float), got negative value '{SNR_value}' instead."
            raise ValueError(err_msg)

        # Cast SNR_value to a float scalar tensor and return it
        return torch.tensor(float(SNR_value))
    
    def set_gamma_scale(self, gamma_scale):
        """
        Set the 'gamma(t)' scale factor.
        
        Args:
            gamma_scale (float or int): Scale factor for 'gamma(t)'
            
        Return:
            None
        
        """
        ## 1) Check that 'gamma_scale' is a strictly positive number 
        ##    (int or float) and cast it to float in any case
        # 1a) Check that it is a number
        if not isinstance(gamma_scale, (int, float)):
            err_msg = f"The input 'gamma_scale' must be a strictly positive number (int or float), got type '{type(gamma_scale)}' instead."
            raise TypeError(err_msg)
    
        # 1b) Enforce strict positivity
        if gamma_scale<=0:
            err_msg = f"The input 'gamma_scale' must be a strictly positive number (int or float), got non-strictly-positive value '{gamma_scale}' instead."
            raise ValueError(err_msg)
        
        # 1c) Cast to float
        gamma_scale = float(gamma_scale)

        # 2) Assign gamma scale to the corresponding class attribute
        self.gamma_scale = gamma_scale

        # 3) Set the gamma parameter (that depend on self.gamma_scale)
        self.set_gamma_params()

    def set_gamma_params(self):
        """ Set the parameters of gamma(t). """
        # Use SNR(t) = exp( -gamma(t) ) <=> gamma(t) = -log( SNR(t) ) to compute 
        # the minimal and maximal gamma values based on the max and min SNRs
        gamma_min = -torch.log(self.SNR_max)
        gamma_max = -torch.log(self.SNR_min)

        # Define the parameters of gamma(t) as defined in the article 
        # 'Classifier Free Diffusion Guidance'
        # (https://arxiv.org/pdf/2207.12598.pdf) by J. Ho & T. Salimans
        # where 'gamma(t)' corresponds to '-lambda(u)' and thus 
        # gamma_min=-lamba_max and gamma_max=-lambda_min.
        # Remark: self.gamma_scale=2 in the article.
        self._gamma_param_b = torch.arctan(torch.exp(gamma_min/self.gamma_scale))
        self._gamma_param_a = torch.arctan(torch.exp(gamma_max/self.gamma_scale))-self._gamma_param_b

    def gamma(self, t):
        """
        Return gamma(t) for t in [0, 1]. 
        
        Remark:
        Use gamma(t) as defined in the article 'Classifier Free Diffusion Guidance'
        (https://arxiv.org/pdf/2207.12598.pdf) by J. Ho & T. Salimans
        where 'gamma(t)' corresponds to '-lambda(u)' in the article.

        """
        return self.gamma_scale*torch.log( torch.tan(self._gamma_param_a*t+self._gamma_param_b) )

    def deriv_gamma(self, t):
        """
        Return derivative of gamma(t) for t in [0, 1].
        
        Remark:
        Use gamma(t) as defined in the article 'Classifier Free Diffusion Guidance'
        (https://arxiv.org/pdf/2207.12598.pdf) by J. Ho & T. Salimans
        where 'gamma(t)' corresponds to '-lambda(u)' in the article.

        """
        return self.gamma_scale*self._gamma_param_a/torch.sin(self._gamma_param_a*t+self._gamma_param_b)/torch.cos(self._gamma_param_a*t+self._gamma_param_b)

    def time_weight(self, t):
        """ Return the time-dependent weights for the loss. """
        # Differ cases 
        if self.time_weighting_strategy=='uniform':
            # Use uniform weights for the loss (i.e. actually do not use any weight)
            return torch.ones_like(t).squeeze()
        elif self.time_weighting_strategy=='derivative_of_gamma':
            # The actual weight of the diffusion loss corresponds to the derivative
            # of 'gamma(t)' at the input times 't'
            return self.deriv_gamma(t).squeeze()
        else:
            err_msg = f"The time weighting strategy '{self.time_weighting_strategy}' has not been implemented."
            raise NotImplementedError(err_msg)

    def kappa2(self, t):
        """
        Return kappa^2(t) for t in [0, 1]. 
        
        Remark:
        Use kappa^2(t) as defined in the article 'Variational Diffusion Models'
        (https://arxiv.org/pdf/2107.00630.pdf) by D. Kingma, T. Salimans, 
        B. Poole, and J. Ho where 'kappa^2(t)' corresponds to 'alpha^2(t)' in 
        the article.
        
        """
        return self.sigmoid_fn(-self.gamma(t))

    def sigma2(self, t):
        """
        Return sigma^2(t) for t in [0, 1]. 
        
        Remark:
        Use sigma^2(t) as defined in the article 'Variational Diffusion Models'
        (https://arxiv.org/pdf/2107.00630.pdf) by D. Kingma, T. Salimans, 
        B. Poole, and J. Ho.
        
        """
        return self.sigmoid_fn(self.gamma(t))

    def SNR(self, t):
        """
        Return 'signal-to-noise ratio' SNR(t) for t in [0, 1]. 
        
        Remark:
        Use SNR(t) as defined in the article 'Variational Diffusion Models'
        (https://arxiv.org/pdf/2107.00630.pdf) by D. Kingma, T. Salimans, 
        B. Poole, and J. Ho.
        
        """
        return self.exp(-self.gamma(t))
    
    def phi(self, t):
        """
        One can identify kappa(t)=exp(-[phi(t)-phi(0)]) where phi
        is the function used to claculate the transition matrix:
        Q(z_t|z_s)=[ exp(R_b[phi(t)-phi(s)]) ]_{z_t, z_s}
        with R_b as the constant 'base' rate matrix.

        # Thus,
        phi(t) - phi(0) = -log(kappa(t)) 
        phi(t) - phi(0) = -1/2*log(kappa^2(t))
        phi(t) - phi(0) = -1/2*log( Sigmoid(-gamma(t)) )
                        = ...
        phi(t) - phi(0) =  1/2*log( 1+exp(gamma(t)) )

        As only differences in phi (for different times) occur, phi
        has a shift-invariance, i.e. phi(t) = phi_offset + f(t) s.t.
        phi(t) - phi(s) = f(t) - f(s).
        Obviously, this does not affect derivatives, i.e. phi'(t) = f'(t).
        
        Demanding that phi(0)=0, we get (including this aformenetioned shift)
        phi(t) = phi_offset + 1/2*log( 1+exp(gamma(t)) )
        and thus [because we demand that phi(0)=0]
        phi_offset = -1/2*log( 1+exp(gamma(0)) )

        Remarks:
        (1) Note that in this definition, the time dependent (instantaneous) 
            rate matrix is then given by R_{t} = R_b*phi'(t) at time t over 
            the derivative of phi.

        (2) In the article 'A Continuous Time Framework for Discrete Denoising Models'
            (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al., phi'(t) is
            defined as beta(t).

        """
        # Determine phi_offset [that ensures phi(t=0)=0 for phi(t)=1/2*log( 1+exp(gamma(t)) )]
        phi_offset = -1/2*torch.log( 1 + torch.exp(self.gamma(0)) )

        # Return phi(t) = phi_offset + 1/2*log( 1 + exp(gamma(t)) )
        return phi_offset + 1/2*torch.log( 1 + torch.exp(self.gamma(t)) )
    
    def deriv_phi(self, t):
        """
        One can identify kappa(t)=exp(-[phi(t)-phi(0)]) where phi
        is the function used to claculate the transition matrix:
        Q(z_t|z_s)=[ exp(R_b[phi(t)-phi(s)]) ]_{z_t, z_s}
        with R_b as the constant 'base' rate matrix.

        Once can show (see method 'phi') that:
        phi(t) = phi_0 + 1/2*log( 1+exp(gamma(t)) )

        Thus,
        phi'(t) = 1/2*[ exp(gamma(t))/(1+exp(gamma(t))) ]*gamma'(t)
                = 1/2*Sigmoid(gamma(t))*gamma'(t)
                = 1/2*sigma^2(t)*gamma'(t)

        Remark:
        In the article 'A Continuous Time Framework for Discrete Denoising Models'
        (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al., phi'(t) is
        defined as beta(t).

        """
        return 1/2*self.sigmoid_fn(self.gamma(t))*self.deriv_gamma(t)

    def show_noise_schedule(self):
        # Define the color scheme and other plot specs
        color_dict = {
            'gamma': 'black',
            'deriv_gamma': 'orange',
            'phi': 'tan',
            'deriv_phi': 'teal',
            'kappa': 'r',
            'sigma': 'b',
            'kappa2_ratio': 'forestgreen',
        }
        plot_alpha = 1
        plot_lw    = 2

        # Create an array of time values
        t_plot = torch.linspace(0, 1, 250)

        # Make the figure
        fig, axs = plt.subplots(3, 1, figsize=(5, 15) )

        ## Subplot 1:
        ## Show kappa, sigma, kappa^2, and sigma^2
        ax = axs[0]

        # Show horizontal lines at 0 and 1 in y from 0 to 1 in x
        ax.hlines(0, 0, 1, ls='-', color='grey', lw=0.5, alpha=0.5)
        ax.hlines(1, 0, 1, ls='-', color='grey', lw=0.5, alpha=0.5)

        # Show kappa, sigma, kappa^2, and sigma^2
        kappa2_t_plot = self.kappa2(t_plot)
        sigma2_t_plot = self.sigma2(t_plot)
        ax.plot(t_plot, torch.sqrt(kappa2_t_plot), '-', color=color_dict['kappa'], alpha=plot_alpha, lw=plot_lw, label=r'$\kappa(t)$')
        ax.plot(t_plot, kappa2_t_plot, '--', color=color_dict['kappa'], alpha=plot_alpha, lw=plot_lw, label=r'$\kappa^{2}(t)$')
        ax.plot(t_plot, torch.sqrt(sigma2_t_plot), '-', color=color_dict['sigma'], alpha=plot_alpha, lw=plot_lw, label=r'$\sigma(t)$')
        ax.plot(t_plot, sigma2_t_plot, '--', color=color_dict['sigma'], alpha=plot_alpha, lw=plot_lw, label=r'$\sigma^{2}(t)$')

        # Set plot specs
        ax.set_xlabel(r'$t$')
        ax.legend()
        ax.set_xlim([0, 1])


        ## Subplot 2:
        ## Show the gamma and the derivative of gamma
        ax = axs[1]

        # Show horizontal lines at 0 in y from 0 to 1 in x
        ax.hlines(0, 0, 1, ls='-', color='grey', lw=0.5, alpha=0.5)

        # Show the gamma, the derivative of gamma, and the ratio of kappa2
        ax.plot(t_plot, self.gamma(t_plot), '-', color=color_dict['gamma'], alpha=plot_alpha, lw=plot_lw, label=r'$\gamma(t)$')
        ax.plot(t_plot, self.deriv_gamma(t_plot), '-', color=color_dict['deriv_gamma'], alpha=plot_alpha, lw=plot_lw, label=r'$\gamma^{\,\prime}(t)$')

        # Determine the ratio kappa2(t_{n})/kappa2(t_{n-1})
        kappa2_ratio_t_plot = kappa2_t_plot[1:]/kappa2_t_plot[:-1]
        ax.plot(t_plot[:-1], kappa2_ratio_t_plot, '-', color=color_dict['kappa2_ratio'], label=r'ratio-$\kappa^{2}(t)$')

        # Set plot specs
        ax.set_xlabel(r'$t$')
        ax.legend()
        ax.set_xlim([0, 1])

        
        ## Subplot 3:
        ## Show the gamma and the derivative of gamma
        ax = axs[2]

        # Show horizontal lines at 0 in y from 0 to 1 in x
        ax.hlines(0, 0, 1, ls='-', color='grey', lw=0.5, alpha=0.5)

        # Show the phi and the derivative of phi
        phi_t_plot       = self.phi(t_plot)
        deriv_phi_t_plot = self.deriv_phi(t_plot)
        ax.plot(t_plot, phi_t_plot, '-', color=color_dict['phi'], alpha=plot_alpha, lw=plot_lw, label=r'$\varphi(t)$')
        ax.plot(t_plot, deriv_phi_t_plot, '-', color=color_dict['deriv_phi'], alpha=plot_alpha, lw=plot_lw, label=r'$\varphi^{\,\prime}(t)$')

        # Set plot specs
        ax.set_xlabel(r'$t$')
        ax.legend()
        ax.set_xlim([0, 1])


        ## Set global plot specs
        plt.tight_layout()
        plt.show()

        # Print out some information about the noise schedule
        print('-'*100)
        print("Noise schedule information:")
        print('-'*100)
        print(f"SNR_min:   {self.SNR_min}")
        print(f"SNR_max:   {self.SNR_max}")
        print(f"kappa_min: {torch.min(torch.sqrt(kappa2_t_plot))}")
        print(f"kappa_max: {torch.max(torch.sqrt(kappa2_t_plot))}")
        print(f"sigma_min: {torch.min(torch.sqrt(sigma2_t_plot))}")
        print(f"sigma_max: {torch.max(torch.sqrt(sigma2_t_plot))}")
        print(f"kappa2_ratio_min: {torch.min(kappa2_ratio_t_plot)}")
        print(f"kappa2_ratio_max: {torch.max(kappa2_ratio_t_plot)}")
        print(f"phi_min: {torch.min(phi_t_plot)}")
        print(f"phi_max: {torch.max(phi_t_plot)}")
        print(f"deriv_phi_min: {torch.min(deriv_phi_t_plot)}")
        print(f"deriv_phi_max: {torch.max(deriv_phi_t_plot)}")
        print('-'*100)
        print()

    def set_random_seed(self, random_seed):
        """ Set random seed(s) for reproducibility. """
        # Set random seeds for any modules that potentially use randomness
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)

    def train(self, train_dataloader, which_model='denoising', valid_dataloader=None, num_epochs=10, random_seed=42):
        """
        Train the model specified by 'which_model'.
        
        Args:
            train_dataloader (torch.utils.data.Dataloader): Dataloader for the 
                iteration over batches of the training dataset.
            which_model (str): Which model to train 'denoising' or 'property'.
            valid_dataloader (torch.utils.data.Dataloader or None): If not None,
                dataloader for the iteration over batches of the validation 
                dataset.
                (Default: None)
            num_epochs (int): Number of epochs to train for.
                (Default: 10)
            random_seed(int): Random seed to be used for training.
                (Default: 42)
        
        Return:
            None
        
        """
        # Check that the dataset of the train dataloader is of type 'DictDataset'
        if not isinstance(train_dataloader.dataset, utils.DictDataset):
            err_msg = f"The input 'train_dataloader' must be of type 'DictDataset', got type '{type(train_dataloader)}' instead."
            raise TypeError(err_msg)

        # Check that the dataset of the validation dataloader is of type 'DictDataset' (if it is not None)
        if valid_dataloader is not None:
            if not isinstance(valid_dataloader.dataset, utils.DictDataset):
                err_msg = f"The input 'valid_dataloader' must be of type 'DictDataset', got type '{type(valid_dataloader)}' instead."
                raise TypeError(err_msg)

        # Set a random seed
        self.set_random_seed(random_seed)

        # Set the number of x-features to None
        self.x_num_features = None

        # Notify user
        print(f"Training the '{which_model}' model for '{num_epochs}' epochs.")

        # Loop over the epochs
        epoch_loss_list= list()
        for epoch in range(num_epochs):
            batch_loss_list = list()
            for batch_data in train_dataloader:
                # Train on the batch for the specified model
                batch_loss_value = self.train_on_batch(batch_data, which_model)

                # Add the loss to the batch loss list
                batch_loss_list.append(batch_loss_value)

            # Take the sum over the losses in the batch thereby determining
            # the total loss of the current epoch and append this to the epoch
            # loss list
            epoch_loss = np.sum(np.array(batch_loss_list))
            epoch_loss_list.append(epoch_loss)

            # Print the current loss
            if epoch%100==0:
                print(f"[{epoch}] {epoch_loss}")

        # Plot the losses of the epochs
        plt.figure(figsize=(6, 6))
        plt.plot(epoch_loss_list, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (rolling)')
        y_min = min([0, np.min(epoch_loss_list)*1.1])
        plt.ylim([y_min, np.mean(epoch_loss_list[1:])*3])
        plt.show()
        print(f"Last (rolling epoch) loss: {epoch_loss_list[-1]}")

    def train_on_batch(self, batch_data, which_model):
        """"
        Return the loss over the input batch data.
        
        Args:
            batch_data (torch.utils.data.Dataset): Torch dataset object
                holding the data of the current batch that can be passed
                to the model specified by the input 'which_model'.
            which_model (str): For which model should we calculate the
                batch loss? Expected is 'denoising' or 'property'.

        Return:
            (float): Loss value.

        """
        # Check that the model specified by 'which_model' is defined.
        # Thus, check if 'which_model' is a dictionary-key of self.model_dict 
        # and the corresponding dictionary-value (i.e. the model) is not None.
        if (which_model not in self.model_dict) or (self.model_dict[which_model] is None):
            err_msg = f"No model with name '{which_model}' has been defined."
            raise ValueError(err_msg)

        # Zero the gradients
        self.optim_dict[which_model].zero_grad()

        # Determine the loss over the batch 
        batch_loss = self.get_batch_loss(batch_data, which_model)

        # Backpropagate and thus set the gradients of the model parameters
        # w.r.t. the loss
        batch_loss.backward()

        # Perform model parameter updates
        self.optim_dict[which_model].step()

        # Return the loss value
        return batch_loss.item()

    def get_batch_loss(self, batch_data, which_model):
        """
        Return the loss over the input batch data.
        
        Args:
            batch_data (torch.utils.data.Dataset): Torch dataset object
                holding the data of the current batch that can be passed
                to the model specified by the input 'which_model'.
            which_model (str): For which model should we calculate the
                batch loss? Expected is 'denoising' or 'property'.

        Return:
            (torch.tensor): Scalar loss of the batch for the passed batch data.
        
        """
        # Differ cases for 'which_model'
        if which_model=='denoising':
            return self.diffusion_batch_loss(batch_data)
        elif which_model=='property':
            return self.property_batch_loss(batch_data)
        else:
            err_msg = f"The batch loss is only define for 'which_model' corresponding to 'denoising' or 'property'."
            raise ValueError(err_msg)

    def property_batch_loss(self, batch_data):
        """
        Return the batch loss for the property model. 
        
        Args:
            batch_data (torch.utils.data.Dataset): Torch dataset object
                holding the data of the current batch that can be passed
                to the denoising model.

        Return:
            (torch.tensor): Scalar loss of the batch for the passed batch data.
        
        """
        # Access the batch x-data
        batch_x = batch_data['x']
        
        # Access the y-data
        if 'y' not in batch_data:
            err_msg = f"To train the property model, the batch data must include 'y' values, but didn't."
            raise KeyError(err_msg)
        
        batch_y = batch_data['y']

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
        # Remark: 'self.forward_sample_z_t' returns the state and the noise 'epsilon'
        batch_z_t = self.forward_sample_z_t(batch_x, batch_t)


        # Set the property model into 'train mode'
        self.model_dict['property'].train()
        
        # Determine the log-probability (i.e. log-likelihood) of the property 
        # model for the batch data for each point in the batch, i.e. log_prob 
        # is a 1D torch tensor of shape (batch_size,)
        log_prob = self.model_dict['property'].log_prob(batch_z_t, batch_t, batch_y)

        # The loss is given by the negative log-probability summed over the 
        # entire batch
        loss = -torch.sum(log_prob)

        return loss
    
    def _sample_times_over_batch(self, batch_size):
        """
        Sample a time t in [0, 1] for each point in the batch.

        Args:
            batch_size (int): Batch size, i.e. number of points 
                for which a time should be sampled for)

        Return:
            (torch.tensor): Sampled times as 2D torch tensor of shape
                (batch_size, 1) containing times in [0, 1].
        """
        # I.i.d. draw batch_size many samples from [0, 1] and obtaining
        # a torch tensor of shape (batch_size, 1)
        batch_t = torch.rand(batch_size, 1)
        #print(f"batch_t.shape: {batch_t.shape}")
        #print(f"batch_t: {batch_t[:10]}")

        return batch_t


    def _parse_time(self, t, batch_size):
        """
        Parse the input time 't' using the input 'x' as 'template'.

        Args:
            t (torch.tensor, float or int): Time to be parsed either as number,
                as 2D torch tensor of shape (batch_size, 1), or as 1D torch
                tensor of shape (batch_size, )
            batch_size (int): Batch size.

        Returns:
            (torch.tensor): The time as 2D torch tensor of shape (batch_size, 1).
        """
        # Differ cases for type of input t
        if torch.is_tensor(t):
            # Differ cases for dimension of time t
            if t.dim()==1:
                # Check that time t has the correct shape
                if t.shape[0]!=batch_size:
                    err_msg = f"If the time is passed as 1D torch tensor it must have shape (batch_size,), got '{t.shape}' instead."
                    raise ValueError(err_msg)

                # If the checks above were fine, reshape time to a 2D tensor of shape (batch_size, 1)
                # and assign it to the parsed time
                parsed_t = t.reshape(-1, 1)

            elif t.dim()==2:
                # Check that time t has the correct shape
                if not (t.shape[0]==batch_size and t.shape[1]==1):
                    err_msg = f"If the time is passed as 2D torch tensor it must have shape (batch_size, 1), got '{t.shape}' instead."
                    raise ValueError(err_msg)

                # If the checks above were fine, assign time to the parsed time
                parsed_t = t

            else:
                err_msg = f"If the time is passed as torch tensor it must be either a 2D tensor of shape (batch_size, 1) or a 1D tensor of shape (batch_size,), got dimension '{t.dim()}' instead."
                raise ValueError(err_msg)
            
        elif isinstance(t, (int, float)):
            # Parse the time as a torch tensor
            parsed_t = t*torch.ones(batch_size, 1)
        else:
            err_msg = f"The input time 't' must be either a number (float or int) or a torch.tensor of shape (batch_size, ) where the expected batch size is '{batch_size}'."
            raise TypeError(err_msg)

        return parsed_t
    
    ##############################################################################################################
    ### SPECIFY METHODS THAT HAVE TO BE IMPLEMENTED BY DERIVED CLASSES
    ##############################################################################################################
    
    def diffusion_batch_loss(self, *args, **kwargs):
        """
        Return the batch loss for diffusion.         
        """
        raise NotImplementedError("The method 'diffusion_batch_loss' must be implemented by any derived class.")
    
    def forward_sample_z_t(self, *args, **kwargs):
        """
        Sample z_t in forward diffusion (i.e. noised state at time t).
        """
        raise NotImplementedError("The method 'forward_sample_z_t' must be implemented by any derived class.")
    
    def generate(self, *args, **kwargs):
        """
        Generate 'novel' points by sampling from p(z_1) and then using ancestral 
        sampling (backwards in time) to obtain 'novel' \hat{z}_0=\hat{x}.
        """
        raise NotImplementedError("The method 'generate' must be implemented by any derived class.")

    def _backward_sample_z_t_next(self, *args, **kwargs):
        """
        Sample next state backward in time from last (i.e. current) state.
        """
        raise NotImplementedError("The method '_backward_sample_z_t_next' must be implemented by any derived class.")
    
    def _sample_z_1(self, *args, **kwargs):
        """
        I.i.d. drawn z_1 for each point in a batch. 
        """
        raise NotImplementedError("The method '_sample_z_1' must be implemented by any derived class.")
