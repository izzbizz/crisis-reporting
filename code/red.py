import pandas as pd
import numpy as np
import random

from scipy.stats import norm, multinomial

class RED:
    def __init__(self, X, k, smoothing=0.000001):
        self.k = k        # number of topics
        self.M = len(X)   # number of articles
        self.X = X

        persons = []
        for liste in X.person_vector:
            persons.append(np.array(liste))
        self.persons = np.array(persons)

        places = []
        for liste in X.place_vector:
            places.append(np.array(liste))
        self.places = np.array(places)

        keywords = []
        for liste in X.general_vector:
            keywords.append(np.array(liste))
        self.keywords = np.array(keywords)

        self.days = X.day.values
        self.N_k = len(self.keywords[0])
        self.N_l = len(self.places[0])
        self.N_p = len(self.persons[0])
       
        self.a = smoothing  # Laplace smoothing
    
        # initialize posterior (matrix: probabilities of event per article)
        posterior = np.random.rand(self.M, self.k)
        self.posterior = posterior / posterior.sum(axis=1)[:, None]
        
        # initialize pi (event probabilities)
        pi = np.random.uniform(size=self.k)
        self.pi = pi / pi.sum()
        
        # initialize phi (emission probabilities)
        # keyword probabilities
        self.phi_k = np.empty((self.N_k, self.k))
        for e in range(0, self.k):
            em = np.random.uniform(size=self.N_k)
            em = em / np.sum(em)
            self.phi_k[:, e] = em
        
        # location probabilities
        self.phi_l = np.empty((self.N_l, self.k))
        for e in range(0, self.k):
            em = np.random.uniform(size=self.N_l)
            em = em / np.sum(em)
            self.phi_l[:, e] = em
    	
    	# person probabilities
        self.phi_p = np.empty((self.N_p, self.k))
        for e in range(0, self.k):
            em = np.random.uniform(size=self.N_p)
            em = em / np.sum(em)
            self.phi_p[:, e] = em
		
		# initialize gmm parameters        
        self.mus = random.sample(range(1, 30), self.k)
        self.sigmas = random.sample(range(1, 7), self.k)
    
    def fit(self, tol=.1, max_iters=10):
        num_iters = 0
        ll = 0
        previous_ll = 1
        while (num_iters < max_iters) and (ll + tol < previous_ll): 
            previous_ll = ll
            self._fit()
            num_iters += 1
            ll = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
    
    def loglikelihood(self):
        ll = 0
        for x in range(self.M):
            tmp = 0
            for e in range(self.k):
                t_prob = norm.pdf(self.days[x], self.mus[e], self.sigmas[e]) + self.a
                k_prob = multinomial.pmf(self.keywords[x], len(self.keywords[x]), self.phi_k[:, e]) + self.a
                p_prob = multinomial.pmf(self.persons[x], len(self.persons[x]), self.phi_p[:, e]) + self.a
                l_prob = multinomial.pmf(self.places[x], len(self.places[x]), self.phi_l[:, e]) + self.a
                tmp += (self.pi[e] + self.a) * k_prob * p_prob * l_prob * t_prob
#                print((self.pi[e] + self.a) * k_prob * p_prob * l_prob * t_prob)
            ll -= np.log(tmp)
        return ll

    def _fit(self):
        self.e_step()
        self.m_step()
            
    def e_step(self):
        person_probs = np.matmul(self.persons, self.phi_p) + self.a
        place_probs = np.matmul(self.places, self.phi_l) + self.a
        keyw_probs = np.matmul(self.keywords, self.phi_k) + self.a
        
        time_probs = []
        for e in range(self.k):
            time_probs.append(np.array([norm.pdf(day, self.mus[e], self.sigmas[e]) + self.a for day in self.days]))
        time_probs = np.array(time_probs).T
        
        probs = np.multiply(person_probs, place_probs)
        probs = np.multiply(probs, keyw_probs)
        probs = np.multiply(probs, time_probs)
        probs = np.multiply(probs, np.reshape(self.pi, (1, self.k)))
        self.posterior = np.divide(probs, np.sum(probs, axis=1).reshape(-1,1))
            
    def m_step(self):
        self._mstep_gaussian()
        self._mstep_multi(self.phi_k, self.keywords, self.N_k)
        self._mstep_multi(self.phi_l, self.places, self.N_l)
        self._mstep_multi(self.phi_p, self.persons, self.N_p)
        self._update_pi()
   
    def _mstep_gaussian(self):
        for e in range(self.k):
            self.mus[e] = np.sum(self.posterior[:, e] * self.days) / np.sum(self.posterior[:, e])
            self.sigmas[e] = np.sum(self.posterior[:, e] * (self.days - self.mus[e])**2) / \
                                np.sum(self.posterior[:, e])

    def _mstep_multi(self, probs, data, N):
        for e in range(self.k):
            probs[:, e] = [(1 + np.sum([self.posterior[x, e] * vec[w] for x, vec in enumerate(data)])) /\
                           (N + np.sum([self.posterior[x, e] * np.sum(vec) for x, vec in enumerate(data)])) \
                           for w in range(N)]
                                   
    def _update_pi(self):
        for e in range(self.k):
            self.pi[e] = np.sum(self.posterior[:, e]) / self.M

    def predict(self, article):
        pass
