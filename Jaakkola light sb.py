import math

from torch import nn
import numpy as np

from torch.nn.functional import softmax, log_softmax
import torch
import time

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from numpy.polynomial import Polynomial



class LightSBplus(nn.Module):
    def __init__(self, dim=2, n_potentials=5, epsilon=1,
                 sampling_batch_size=1, batch_size=1, S_diagonal_init=0.1):
        super().__init__()
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size
        
#         self.log_alpha_raw = nn.Parameter(self.epsilon*torch.log(torch.ones(n_potentials)/n_potentials))
        self.log_alpha_raw = torch.log(torch.ones(n_potentials)/n_potentials)
#         self.r = nn.Parameter(torch.randn(n_potentials, dim))
        self.r = torch.randn(n_potentials, dim)
        
#         self.log_S = nn.Parameter(torch.log(S_diagonal_init*torch.ones(n_potentials, self.dim)))
        self.log_S = torch.log(S_diagonal_init*torch.ones(n_potentials, self.dim))
        
        self.sum_component_probs = torch.zeros(n_potentials)
        self.n_samples = 0

        self.gm = torch.ones(batch_size) * 1000
        
    def reset_stats(self):  #НЕ используется?
        print("reset_stats: Oh, no, i'm useful!!!")

        self.sum_component_probs = torch.zeros_like(self.sum_component_probs)
        self.n_samples = 0
        
    def init_r_by_samples(self, samples):
        assert samples.shape[0] == self.r.shape[0]
        
        self.r.data = torch.clone(samples.to(self.r.device))
    
    def init_raw_log_alpha(self, x): #НЕ используется?
        print("init_raw_log_alpha: Oh, no, i'm useful!!!")

        x = x[None, :]
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        
        eps_S = epsilon*S
                    
        x_S_x = (x[:, None, :]*S[None, :, :]*x[:, None, :]).sum(dim=-1)
        x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
        
        self.log_alpha_raw.data = -0.5*(x_S_x + 2*x_r)[0]
        
    
    def get_S(self):
        S = torch.exp(self.log_S)
        return S
    
    def get_r(self):
        return self.r

    def get_gm(self):
        return self.gm
    
    def get_log_alpha(self):
        return self.log_alpha_raw
    
        
    @torch.no_grad()
    def forward(self, x): #Менять не нужно
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        
        log_alpha = self.get_log_alpha()
        
        eps_S = epsilon*S
            
        samples = []
        batch_size = x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size//sampling_batch_size if batch_size % sampling_batch_size == 0 else (batch_size//sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = x[sampling_batch_size*i:sampling_batch_size*(i+1)]
            
            x_S_x = (sub_batch_x[:, None, :]*S[None, :, :]*sub_batch_x[:, None, :]).sum(dim=-1)
            x_r = (sub_batch_x[:, None, :]*r[None, :, :]).sum(dim=-1)
            r_x = r[None, :, :] + S[None, :]*sub_batch_x[:, None, :]
                
            exp_argument = (x_S_x + 2*x_r)/(2*epsilon) + log_alpha[None, :]
                     
            mix = Categorical(logits=exp_argument)
            comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon*S)[None, :, :]), 1)
            gmm = MixtureSameFamily(mix, comp)

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)

        return samples
        
    
    def get_drift(self, x, t):
        x = torch.clone(x)
        x.requires_grad = True
        
        epsilon = self.epsilon
        r = self.get_r()
        
        S_diagonal = torch.exp(self.log_S) # shape: potential*dim
        A_diagonal = (t/(epsilon*(1-t)))[:, None, None] + 1/(epsilon*S_diagonal)[None, :, :] # shape: batch*potential*dim
        
        S_log_det = torch.sum(self.log_S, dim=-1) # shape: potential
        A_log_det = torch.sum(torch.log(A_diagonal), dim=-1) # shape: batch*potential
        
        log_alpha = self.get_log_alpha() # shape: potential
        
        S = S_diagonal # shape: potential*dim
        A = A_diagonal # shape: batch*potential*dim

        S_inv = 1/S # shape: potential*dim
        A_inv = 1/A # shape: batch*potential*dim

        c = ((1/(epsilon*(1-t)))[:, None]*x)[:, None, :] + (r/(epsilon*S_diagonal))[None, :, :] # shape: batch*potential*dim

        exp_arg = (
            log_alpha[None, :] - 0.5*S_log_det[None, :] - 0.5*A_log_det
            - 0.5*((r*S_inv*r)/epsilon).sum(dim=-1)[None, :] + 0.5*(c*A_inv*c).sum(dim=-1)
        )

        lse = torch.logsumexp(exp_arg, dim=-1)
        drift = (-x/(1-t[:, None]) + epsilon*torch.autograd.grad(lse, x, grad_outputs=torch.ones_like(lse, device=lse.device))[0]).detach()
        
        return drift
    
    
    def sample_euler_maruyama(self, x, n_steps): #НЕ используется?
        print("sample_euler_maruyama: Oh, no, i'm useful!!!")

        epsilon = self.epsilon
        t = torch.zeros(x.shape[0], device=x.device)
        dt = 1/n_steps
        trajectory = [x]
        
        for i in range(n_steps):
            x = x + self.get_drift(x, t)*dt + math.sqrt(dt)*torch.sqrt(epsilon)*torch.randn_like(x, device=x.device)
            t += dt
            trajectory.append(x)
            
        return torch.stack(trajectory, dim=1)
    
    
    def sample_at_time_moment(self, x, t): #НЕ используется?
        print("sample_at_time_moment: Oh, no, i'm useful!!!")

        t = t.to(x.device)
        y = self(x)
        
        return t*y + (1-t)*x + np.sqrt(t*(1-t)*self.epsilon)*torch.randn_like(x)
    
    
    def get_log_potential(self, x): #Логирование, менять не нужно
        S = self.get_S()
        r = self.get_r()
        log_alpha = self.get_log_alpha()
        D = self.dim
        epsilon = self.epsilon

        #if (S != S).any():
        #    print("None error")

        #if (S < 0).any():
        #    print("<0 error")
        
        mix = Categorical(logits=log_alpha)
        comp = Independent(Normal(loc=r, scale=torch.sqrt(self.epsilon*S)), 1)
        gmm = MixtureSameFamily(mix, comp)

        potential = gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)
        
        return potential
    
    
    def get_log_C(self, x): #Логирование, менять не нужно
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.get_log_alpha()
        
        eps_S = epsilon*S
        
        x_S_x = (x[:, None, :]*S[None, :, :]*x[:, None, :]).sum(dim=-1)
        x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
        
        exp_argument = (x_S_x + 2*x_r)/(2*epsilon) + log_alpha[None, :]
        
        self.sum_component_probs += torch.nn.functional.softmax(exp_argument, dim=-1).sum(dim=0)
        self.n_samples += x.shape[0]
        
        log_norm_const = torch.logsumexp(exp_argument, dim=-1)
        
        return log_norm_const
    
    
    def set_epsilon(self, new_epsilon):
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)

############################################################################################

    @torch.no_grad()
    def _calc_new_xi(self, x): #НОВОЕ: минимум при xi = eta - gm
        """
        x shape: N x dim

        result shape: N x comp
        """
        eps = self.epsilon
        log_alpha = self.get_log_alpha() # comp
        S = self.get_S() # comp x dim
        r = self.get_r() # comp x dim

        gm = self.get_gm() #N=2000

        # S[None, :, :]*x[:, None, :] shape: N x comp x dim

        a_1 = x.square() @ S.T / (2 * eps)
        a_2 = x @ r.T / eps
        a_3 = log_alpha[None, :]

        return a_1 + a_2 + a_3 - gm[:, None]

    @torch.no_grad()
    def _calc_lambda_xi(self, xi): #НОВОЕ
        """
        xi shape: N x comp

        result shape: N x comp
        """

        a_1 = 1 / (2 * xi + 1e-8)
        a_2 = 1 / (1 + torch.exp(-xi)) - 1 / 2

        lmbd_xi = a_1 * a_2

        return lmbd_xi

    @torch.no_grad()
    def _calc_new_gm(self, x, lmbd_xi): #НОВОЕ
        """
        x shape: N x dim
        lmbd_xi shape: N x comp

        result shape: N
        """

        S = self.get_S()  # comp x dim
        r = self.get_r()  # comp x dim
        log_alpha = self.get_log_alpha()  # comp
        eps = self.epsilon

        K = S.shape[0]

        a = 2 * torch.sum(lmbd_xi, dim=1)

        b1 = -1 + K / 2
        b2 = (1 / eps) * torch.sum((x.square() @ S.T) * lmbd_xi, dim=1)
        b3 = (2 / eps) * torch.sum((x @ r.T) * lmbd_xi, dim=1)
        b4 = 2 * torch.sum(lmbd_xi * log_alpha[None, :], dim=1)

        gm = (b1 + b2 + b3 + b4) / a

        return gm
    
    
    @torch.no_grad()
    def _calc_new_mu(self, y): #НЕ менять, ELBO
        """
        y shape: M x dim
        
        result shape: M x comp
        """
        S = self.get_S() # comp x dim
        r = self.get_r() # comp x dim
        log_alpha = self.get_log_alpha() # comp
        eps = self.epsilon

        D = y.shape[1]
        
#         comp = Independent(Normal(loc=r, scale=torch.sqrt(eps*S)), 1)
        comp = Normal(loc=r[None, :, :].expand(y.shape[0], r.shape[0], r.shape[1]),
                      scale=torch.sqrt(eps*S)[None, :, :].expand(y.shape[0], S.shape[0], S.shape[1]))
#         log_probs = torch.stack([comp.log_prob(elem) for elem in y], dim=0) + log_alpha[None, :] # M x comp

        log_probs = comp.log_prob(y[:, None, :].expand(y.shape[0], r.shape[0], y.shape[1])).sum(dim=-1)
        
#         norm_const = torch.exp(log_alpha).sum()
        
        mu = softmax(log_probs, dim=-1)
        
        return mu
    
    
    @torch.no_grad()
    def _calc_new_l(self): #НЕ менять
        """
        
        result shape: comp x dim
        """
        return torch.clone(self.get_S()).detach()
        
    
    @torch.no_grad()
    def calc_r_loss(self, x, y, mu, xi): #Это вообще не используется????....
        """
        x shape: N x dim
        y shape: M x dim
        mu shape: M x comp
        xi shape: N x comp
        """

        #Дебаг, используем ли функцию
        print("calc_r_loss: Oh, no, i'm useful!!!")

        grad_xi = softmax(xi, dim=-1) # N x comp
        r = self.get_r()
        S = self.get_S() # comp x dim
        S_inv = torch.exp(-self.log_S) # comp x dim
        log_alpha = self.get_log_alpha() # comp
        eps = self.epsilon
        
        a_1 = torch.mean(torch.sum(torch.sum((y[:, None, :] - r[None, :, :]).square()*S_inv[None, :, :], dim=2)*mu, dim=1), dim=0)/(2*eps)
        
#       b_2 = torch.mean((grad_xi - xi)[:, :, None]*x[:, None, :], dim=0) # comp x dim
        a_2 = ((grad_xi - xi)*(x@r.T)).sum(dim=-1).mean(dim=0)/eps
        
        a_3 = ((x@r.T)*(log_alpha[None, :])).sum(dim=-1).mean(dim=0)/eps
        a_4 = torch.mean(torch.sum((x@r.T).square(), dim=1), dim=0)/(2*eps*eps)
        a_5 = torch.mean(torch.sum((x@r.T)*(x.square()@S.T), dim=1), dim=0)/(2*eps*eps)
        
#         pdb.set_trace()
        
        return a_1 + a_2 + a_3 + a_4 + a_5


    @torch.no_grad()
    def _calc_new_r(self, x, y, mu, lmbd_xi): #ИСПРАВИЛА
        """
        x shape: N x dim
        y shape: M x dim
        mu shape: M x comp
        lmbd_xi shape: N x comp

        result shape: comp x dim
        """
        S = self.get_S()  # comp x dim
        S_inv = torch.exp(-self.log_S)  # comp x dim
        log_alpha = self.get_log_alpha()  # comp
        eps = self.epsilon
        gm = self.gm # N

        #а_1 исправила
        #a_1 = (2 / eps) * torch.mean((x[:, :, None] @ x[:, None, :])[:, None, :, :] * lmbd_xi[:, :, None, None], dim=0) # comp x dim x dim
        a_1 = (2 / (eps * x.shape[0])) * torch.einsum('nc,ni,nj->cij', lmbd_xi, x, x)

        # а_2 не менять
        a_2 = torch.diag_embed(S_inv * mu.mean(dim=0)[:, None])  # comp x dim x dim
        a_2 = a_2 + torch.eye(a_2.size(-1), device=a_2.device) * 0.1

        A = a_1 + a_2   # comp x dim x dim

        # S_inv[None, :, :]*y = S^{-1}y: M x comp x dim

        #b_1 не менять
        b_1 = torch.mean(S_inv[None, :, :] * y[:, None, :] * mu[:, :, None], dim=0)  # comp x dim

        #b_2 исправила
        b_2 = 2 * torch.mean(gm[:, None, None] * lmbd_xi[:, :, None] * x[:, None, :], dim=0) # comp x dim

        #b_3 исправила
        b_3 = 2 * torch.mean(lmbd_xi[:, :, None] * log_alpha[None, :, None] * x[:, None, :], dim=0)  # comp x dim

        # S[None, :, :]*x[:, None, :] shape: N x comp x dim

        #b_4 исправила
        b_4 = (1 / eps) * torch.mean((x.square() @ S.T)[:, :, None] * lmbd_xi[:, :, None] * x[:, None, :], dim=0) # comp x dim

        #b_5 добавила
        b_5 = (1 / 2) * torch.mean(x, dim=0) # dim

        #         b = b_1 + b_2 - b_3 - b_4 - b_5 # comp x dim
        b = b_1 + b_2 - b_3 - b_4 - b_5[None, :]

        #         new_r = torch.linalg.solve(A, b) # solve Ar = b => r = A^{-1}b
        A_minus_1 = torch.linalg.inv(A)  # comp x dim x dim
        new_r = (A_minus_1 @ (b[:, :, None]))[:, :, 0]  # => comp x dim x 1

        return new_r

    @torch.no_grad()
    def _calc_new_log_alpha(self, x, mu, lmbd_xi): #ИСПРАВИЛА
        """
        x shape: N x dim
        mu shape: M x comp
        lmbd_xi: N x comp

        result shape: comp
        """
        S = self.get_S() # comp x dim
        r = self.get_r() # comp x dim
        eps = self.epsilon
        gm = self.gm # N

        #не менять
        a_1 = mu.mean(dim=0) # comp

        #исправила
        a_2 = 2 * torch.mean(lmbd_xi * gm[:, None], dim=0) # comp

        #исправила
        a_3 = (2 / eps) * torch.mean((x @ r.T) * lmbd_xi + eps / 4, dim=0) # comp

        #исправила
        a_4 = (1 / eps) * torch.mean((x.square() @ S.T) * lmbd_xi, dim=0) # comp

        #исправила
        b = 2 * torch.mean(lmbd_xi, dim=0) # comp

        new_log_alpha = (a_1 + a_2 - a_3 - a_4) / b

        return log_softmax(new_log_alpha, dim=-1)
        #return new_log_alpha


    def _calc_S_polynomial(self, x, y, mu, lmbd_xi, l): #ИСПРАВИЛА
        """
        x shape: N x dim
        y shape: M x dim
        mu shape: M x comp
        lmbd_xi shape: N x comp
        l shape: comp x dim
        """
        r = self.get_r() # comp x dim
        log_alpha = self.get_log_alpha() # comp
        eps = self.epsilon
        gm = self.gm # N

        comp = mu.shape[1]
        dim = x.shape[1]

        #исправила
        x_3 = torch.mean(torch.norm(x.square(), dim=-1).square()[:, None] * lmbd_xi, dim=0) / (2 * eps * eps) # comp
        x_3 = x_3[:, None].expand(comp, dim) # comp x dim

        #if (x_3 != x_3).any():
        #    print("Nan error x_3")
        #    for i in range(x_3.shape[0]):
        #        for j in range(x_3.shape[1]):
        #            if x_3[i][j] != x_3[i][j]:
        #                print(i, j)
        #                print(x_3[i][j])

        # исправила
        x_2_1 = torch.mean(x.square(), dim=0)[None, :].expand(comp, dim) / (4 * eps)  # comp x dim
        x_2_2 = torch.mean(x.square()[:, None, :] * log_alpha[None, :, None] * lmbd_xi[:, :, None], dim=0) / eps # comp x dim
        x_2_3 = torch.mean(x.square()[:, None, :] * ((x @ r.T) * lmbd_xi)[:, :, None], dim=0) / (eps * eps) # comp x dim
        x_2_4 = torch.mean((x.square()@l.T)[:, :, None] * x.square()[:, None, :] * lmbd_xi[:, :, None], dim=0)/(2 * eps * eps)
        x_2_5 = torch.mean(x.square()[:, None, :] * lmbd_xi[:, :, None] * gm[:, None, None], dim=0) / eps # comp x dim
        x_2_6 = torch.mean(l[None, :, :] * torch.norm(x.square(), dim=-1).square()[:, None, None] * lmbd_xi[:, :, None], dim=0)/(2 * eps * eps) # comp x dim

        x_2 = x_2_1 + x_2_2 + x_2_3 + x_2_4 - x_2_5 - x_2_6 # comp x dim

        #if (x_2 != x_2).any():
        #    print("Nan error x_2")
        #    for i in range(x_2.shape[0]):
        #        for j in range(x_2.shape[1]):
        #            if x_2[i][j] != x_2[i][j]:
        #                print(i, j)
        #                print(x_2[i][j])

        # не менять
        x_1 = mu.mean(dim=0) / 2 # comp
        x_1 = x_1[:, None].expand(comp, dim) # comp x dim

        #if (x_1 != x_1).any():
        #    print("Nan error x_1")
        #    for i in range(x_1.shape[0]):
        #        for j in range(x_1.shape[1]):
        #            if x_1[i][j] != x_1[i][j]:
        #                print(i, j)
        #                print(x_1[i][j])

        # не менять
        x_0 = - torch.mean((y[:, None, :] - r[None, :, :]).square() * mu[:, :, None], dim=0) / (2 * eps) # comp x dim

        #if (x_0 != x_0).any():
        #    print("Nan error x_0")
        #    for i in range(x_0.shape[0]):
        #        for j in range(x_0.shape[1]):
        #            if x_0[i][j] != x_0[i][j]:
        #                print(i, j)
        #                print(x_0[i][j])

        return torch.stack((x_0, x_1, x_2, x_3), dim=0)
    
    
    @torch.no_grad()
    def eval_S_polynomial(self, coefs, values): #НЕ менять, производная полинома в точке
        """
        coefs shape: 4 x comp x dim
        values shape: val x comp x dim
        
        result shape: val x comp x dim
        """
        # d + cx + bx^2 + ax^3
        # dx^-2 + cx^-1 + b + ax
        # -dx^-1 + clog(x) + bx + ax^2/2
        
        return -coefs[0][None, :, :]/values + coefs[1][None, :, :]*torch.log(values) + coefs[2][None, :, :]*values + 0.5*coefs[3][None, :, :]*values.square()
    
    
    @torch.no_grad()
    def _solve_cardano(self, coefs): #НЕ менять, солвер Кардано
        d, c, b, a = coefs[0], coefs[1], coefs[2], coefs[3]

        p = (3*a*c - b.pow(2))/(3*a.pow(2)).type(torch.complex64)
        q = (2*b.pow(3)-9*a*b*c+27*a.pow(2)*d)/(27*a.pow(3)).type(torch.complex64)

        Q = (p/3).pow(3) + (q/2).pow(2)

        alpha = (-q/2 + Q.sqrt()).pow(1/3)
        # beta = (-q/2 - Q.sqrt()).pow(1/3)
        beta = -p/(3*alpha)

        y_1 = alpha + beta
        y_2 = -(alpha+beta)/2 + 1j*(alpha - beta)/2*math.sqrt(3)
        y_3 = -(alpha+beta)/2 - 1j*(alpha - beta)/2*math.sqrt(3)

        y = torch.stack((y_1, y_2, y_3), dim=0)
        x = y - (b/(3*a))[None, ...]
        
        roots = torch.abs(x.real).float() + 1e-5
        
        return roots
    
    
    @torch.no_grad()
    def _solve_numpy(self, coefs): #НЕ менять, численные методы
        roots = []

        for i in range(coefs[0].shape[0]):
            i_roots = []
            for j in range(coefs[0].shape[1]):
                poly = Polynomial(coef=coefs[:, i, j].numpy().tolist())
                i_roots.append(poly.roots().tolist())
            roots.append(i_roots)

        roots = torch.tensor(np.array(roots).real).float().abs() + 1e-5
        roots = roots.permute((2, 0, 1))
        
        return roots
    
    
    @torch.no_grad()
    def _calc_new_S(self, x, y, mu, lmbd_xi, l, solver_type="numpy"): #НЕ менять
        comp = mu.shape[1]
        dim = x.shape[1]
        
        coefs = self._calc_S_polynomial(x=x, y=y, mu=mu, lmbd_xi=lmbd_xi, l=l)
        
        if solver_type == "numpy":
            roots = self._solve_numpy(coefs)
        elif solver_type == "cardano":
            roots = self._solve_cardano(coefs)
        
        losses = self.eval_S_polynomial(coefs, roots)

        argmins = torch.argmin(losses, dim=0)

#         selected_roots = roots[(torch.arange(3)[:, None, None].expand(3, comp, dim) == argmins[None, :, :])].reshape(-1, dim)
        selected_roots = torch.gather(input=roots, dim=0, index = argmins[None, :, :].expand(3, comp, dim))[0, :, :]
        
        return selected_roots
    
    
    @torch.no_grad()
    def make_optimization_step(self, x, y): #добавился параметр gm, lmbd_xi
        xi = self._calc_new_xi(x=x)
        #if (xi != xi).any():
        #    print("None error: xi")

        lmbd_xi = self._calc_lambda_xi(xi=xi)
        #if (lmbd_xi != lmbd_xi).any():
        #    print("None error: lmbd_xi")

        mu = self._calc_new_mu(y=y)
        #if (mu != mu).any():
        #    print("None error: mu")

        l = self._calc_new_l()
        #if (l != l).any():
        #    print("None error: l")

        new_gm = self._calc_new_gm(x=x, lmbd_xi=lmbd_xi)  # тут пиздец
        self.gm = new_gm
        #if (self.gm != self.gm).any():
        #    print("None error: gm")

        new_r = self._calc_new_r(x=x, y=y, mu=mu, lmbd_xi=lmbd_xi)
        self.r = new_r

        #if (self.r != self.r).any():
        #    print("None error: r")

        new_log_alpha = self._calc_new_log_alpha(x=x, mu=mu, lmbd_xi=lmbd_xi)
        self.log_alpha_raw = new_log_alpha

        #if (self.log_alpha_raw != self.log_alpha_raw).any():
        #    print("None error: log_alpha")

        new_S = self._calc_new_S(x=x, y=y, mu=mu, lmbd_xi=lmbd_xi, l=l, solver_type="cardano").float()
        self.log_S = torch.log(new_S)

        #if (self.log_S != self.log_S).any():
        #    print("None error: log_S")
        #    for i in range(self.log_S[0]):
        #        for j in range(self.log_S[1]):
        #            if self.log_S[i][j] != self.log_S[i][j]:
        #                print(i, j)
        #                print(new_S[i][j])
        
    
#     @torch.no_grad()
#     def calc_S_loss(self, S, x, y, mu, xi, l):
#         """
#         S shape: comp x dim
#         x shape: N x dim
#         y shape: M x dim
#         mu shape: M x comp
#         xi shape: N x comp
#         l shape: comp x dim
#         """
#         N = x.shape[0]
#         M = y.shape[0]
        
#         r = self.get_r() # comp x dim
# #         S = self.get_S() # comp x dim
# #         S_inv = torch.exp(-self.log_S) # comp x dim
#         S_inv = 1/S
#         log_alpha = self.get_log_alpha() # comp
#         grad_xi = softmax(xi, dim=-1) # N x comp
#         eps = self.epsilon
        
#         a_1 = torch.mean(0.5*mu*self.log_S.sum(dim=-1)[None, :], dim=0) # comp
        
#         a_2 = torch.mean(mu*torch.sum(S_inv[None, :, :]*(y[:, None, :] - r[None, :, :]).square(), dim=-1), dim=0)/(2*eps) # comp
        
#         a_3 = torch.sum(x.square()@S.T/(2*eps) * (grad_xi - xi))/N
        
#         a_4 = torch.sum(x.square()@S.T/(2*eps) * (log_alpha[None, :] + x@r.T/eps))/N
        
#         a_5 = torch.sum((x.square()@l.T)*(x.square()@S.T)/(4*eps*eps))/N
        
#         a_6 = torch.sum(torch.sum((S - l)*(S-l), dim=-1)[None, :]*x.square().square().sum(dim=-1)[:, None]/(8*eps*eps))/N
        
#         return a_1 + a_2 + a_3 + a_4 + a_5 + a_6
        
        
