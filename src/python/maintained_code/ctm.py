# Since OCTIS no longer installs, this code has been copied here so that we can run the topic model

import datetime
import os
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ContextualInferenceNetwork(nn.Module):
    """Inference Network."""
    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2):
        super(ContextualInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int) or isinstance(output_size, np.int64), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'rrelu':
            self.activation = nn.RReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()

        self.input_layer = nn.Linear(input_size+input_size, hidden_sizes[0])
        self.adapt_bert = nn.Linear(bert_size, hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert):
        """Forward pass."""
        x_bert = self.adapt_bert(x_bert)
        x = self.activation(x_bert)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma

class CombinedInferenceNetwork(nn.Module):
    """Inference Network."""
    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2):
        super(CombinedInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert (isinstance(output_size, int) or isinstance(output_size, np.int64)), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'rrelu':
            self.activation = nn.RReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()

        self.input_layer = nn.Linear(input_size+input_size, hidden_sizes[0])
        self.adapt_bert = nn.Linear(bert_size, input_size)
        self.bert_layer = nn.Linear(hidden_sizes[0], hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert):
        """Forward pass."""
        x_bert = self.adapt_bert(x_bert)
        x = torch.cat((x, x_bert), 1)
        x = self.input_layer(x)

        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma

class DecoderNetwork(nn.Module):
    """AVITM Network."""
    def __init__(self, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, topic_prior_mean=0.0, topic_prior_variance=None):
        super(DecoderNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert (isinstance(n_components, int) or isinstance(n_components, np.int64)) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"
        # and topic_prior_variance >= 0, \
        #assert isinstance(topic_prior_variance, float), \
        #    "topic prior_variance must be type float"

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors

        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation)
        elif infnet == "combined":
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation)
        else:
            raise Exception('Missing infnet parameter, options are zeroshot and combined')
        if torch.cuda.is_available():
            self.inf_net = self.inf_net.cuda()
        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        #self.topic_prior_mean = topic_prior_mean
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)


        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        if topic_prior_variance is None:
            topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

        topic_doc = theta
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            topic_word = self.beta
            # word_dist: batch_size x input_size
            #self.topic_word_matrix = self.beta
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            topic_word = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word, topic_doc

    def get_theta(self, x, x_bert):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert)
            posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta

class CTM(object):
    """Class to train the contextualized topic model"""
    def __init__(
        self, input_size, bert_input_size, inference_type="zeroshot",
        num_topics=10, model_type='prodLDA', hidden_sizes=(100, 100),
        activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
        lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, num_samples=10,
        reduce_on_plateau=False, topic_prior_mean=0.0, top_words=10,
            topic_prior_variance=None, num_data_loader_workers=0):

        assert isinstance(input_size, int) and input_size > 0, \
            "input_size must by type int > 0."
        assert (isinstance(num_topics, int) or isinstance(
            num_topics, np.int64)) and num_topics > 0, \
            "num_topics must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'], \
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in [
            'softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
            'rrelu', 'elu', 'selu'], \
            ("activation must be 'softplus', 'relu', 'sigmoid', 'swish', "
             "'leakyrelu', 'rrelu', 'elu', 'selu' or 'tanh'.")
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(
            momentum, float) and momentum > 0 and momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'], \
            "solver must be 'adam', 'adadelta', 'sgd', 'rmsprop' or 'adagrad'"
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"

        self.input_size = input_size
        self.num_topics = num_topics
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        self.top_words = top_words
        self.bert_size = bert_input_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.topic_prior_mean = topic_prior_mean
        self.topic_prior_variance = topic_prior_variance

        self.model = DecoderNetwork(
            input_size, self.bert_size, inference_type, num_topics,
            model_type, hidden_sizes, activation,
            dropout, self.learn_priors, self.topic_prior_mean,
            self.topic_prior_variance)
        self.early_stopping = EarlyStopping(patience=5, verbose=False)

        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(
                self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        elif self.solver == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.solver == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif self.solver == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=lr, momentum=self.momentum)

        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        self.best_loss_train = float('inf')

        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        self.best_components = None

        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False
        if self.USE_CUDA:
            self.model = self.model.cuda()

    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        KL = 0.5 * (
            var_division + diff_term - self.num_topics + logvar_det_division)
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        loss = KL + RL

        return loss.sum()

    def _train_epoch(self, loader):
        self.model.train()
        train_loss = 0
        samples_processed = 0
        topic_doc_list = []
        for batch_samples in loader:
            X = batch_samples['X']
            X = X.reshape(X.shape[0], -1)
            X_bert = batch_samples['X_bert']
            if self.USE_CUDA:
                X = X.cuda()
                X_bert = X_bert.cuda()

            self.model.zero_grad()
            (prior_mean, prior_variance,
             posterior_mean, posterior_variance, posterior_log_variance,
             word_dists, topic_word, topic_document) = self.model(X, X_bert)
            topic_doc_list.extend(topic_document)

            loss = self._loss(
                X, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)
            loss.backward()
            self.optimizer.step()

            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss, topic_word, topic_doc_list

    def _validation(self, loader):
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            X = batch_samples['X']
            X = X.reshape(X.shape[0], -1)
            X_bert = batch_samples['X_bert']

            if self.USE_CUDA:
                X = X.cuda()
                X_bert = X_bert.cuda()

            self.model.zero_grad()
            (prior_mean, prior_variance,
             posterior_mean, posterior_variance, posterior_log_variance,
             word_dists, topic_word, topic_document) = self.model(X, X_bert)

            loss = self._loss(
                X, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)

            samples_processed += X.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, validation_dataset=None,
            save_dir=None, verbose=True):
        if verbose:
            print("Settings: \n\
                   N Components: {}\n\
                   Topic Prior Mean: {}\n\
                   Topic Prior Variance: {}\n\
                   Model Type: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Dropout: {}\n\
                   Learn Priors: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                self.num_topics, self.topic_prior_mean,
                self.topic_prior_variance, self.model_type,
                self.hidden_sizes, self.activation, self.dropout,
                self.learn_priors, self.lr, self.momentum,
                self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.validation_data = validation_dataset

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)

        train_loss = 0
        samples_processed = 0

        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            s = datetime.datetime.now()
            sp, train_loss, topic_word, topic_document = self._train_epoch(
                train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            if verbose:
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(self.train_data) * self.num_epochs, train_loss, e - s))

            self.best_components = self.model.beta
            self.final_topic_word = topic_word
            self.final_topic_document = topic_document
            self.best_loss_train = train_loss
            if self.validation_data is not None:
                validation_loader = DataLoader(
                    self.validation_data, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_data_loader_workers)
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(
                    validation_loader)
                e = datetime.datetime.now()

                if verbose:
                    print(
                        "Epoch: [{}/{}]\tSamples: [{}/{}]"
                        "\tValidation Loss: {}\tTime: {}".format(
                            epoch + 1, self.num_epochs, val_samples_processed,
                            len(self.validation_data) * self.num_epochs,
                            val_loss, e - s))

                if np.isnan(val_loss) or np.isnan(train_loss):
                    break
                else:
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        if verbose:
                            print("Early stopping")
                        if save_dir is not None:
                            self.save(save_dir)
                        break

    def predict(self, dataset):
        self.model.eval()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_data_loader_workers)

        topic_document_mat = []
        with torch.no_grad():
            for batch_samples in loader:
                X = batch_samples['X']
                X = X.reshape(X.shape[0], -1)
                X_bert = batch_samples['X_bert']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_bert = X_bert.cuda()
                self.model.zero_grad()
                _, _, _, _, _, _, _, topic_document = self.model(X, X_bert)
                topic_document_mat.append(topic_document)

        results = self.get_info()
        results['test-topic-document-matrix'] = np.asarray(
            self.get_thetas(dataset)).T

        return results

    def get_topic_word_mat(self):
        top_wor = self.final_topic_word.cpu().detach().numpy()
        return top_wor

    def get_topic_document_mat(self):
        top_doc = self.final_topic_document
        top_doc_arr = np.array([i.cpu().detach().numpy() for i in top_doc])
        return top_doc_arr

    def get_topics(self):
        assert self.top_words <= self.input_size, "top_words must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        topics_list = []
        if self.num_topics is not None:
            for i in range(self.num_topics):
                _, idxs = torch.topk(component_dists[i], self.top_words)
                component_words = [self.train_data.idx2token[idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
                topics_list.append(component_words)

        return topics_list

    def get_info(self):
        info = {}
        topic_word = self.get_topics()
        topic_word_dist = self.get_topic_word_mat()
        topic_document_dist = self.get_topic_document_mat()
        info['topics'] = topic_word

        info['topic-document-matrix'] = np.asarray(
            self.get_thetas(self.train_data)).T

        info['topic-word-matrix'] = topic_word_dist
        return info

    def _format_file(self):
        model_dir = (
            "AVITM_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_"
            "lr_{}_mo_{}_rp_{}".format(
                self.num_topics, 0.0, 1 - (1. / self.num_topics),
                self.model_type, self.hidden_sizes, self.activation,
                self.dropout, self.lr, self.momentum,
                self.reduce_on_plateau))
        return model_dir

    def save(self, models_dir=None):
        if (self.model is not None) and (models_dir is not None):
            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        epoch_file = "epoch_" + str(epoch) + ".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self.model.load_state_dict(checkpoint['state_dict'])

    def get_thetas(self, dataset):
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)
        final_thetas = []
        for sample_index in range(self.num_samples):
            with torch.no_grad():
                collect_theta = []
                for batch_samples in loader:
                    x = batch_samples['X']
                    x = x.reshape(x.shape[0], -1)
                    x_bert = batch_samples['X_bert']
                    if self.USE_CUDA:
                        x = x.cuda()
                        x_bert = x_bert.cuda()
                    self.model.zero_grad()
                    collect_theta.extend(
                        self.model.get_theta(x, x_bert).cpu().numpy().tolist())

                final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0) / self.num_samples
