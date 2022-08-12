from scipy.stats import entropy as ent
import numpy as np
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon





def score_entropy(stochastic_prediction):

    results=np.zeros((stochastic_prediction.shape[0],stochastic_prediction.shape[2],stochastic_prediction.shape[3]))
    for i in range(stochastic_prediction.shape[0]):
        for j in range(stochastic_prediction.shape[2]):
            for k in range(stochastic_prediction.shape[3]):
                results[i,j,k]= ent(stochastic_prediction[i,:,j,k])
    results = results.reshape(results.shape[0], results.shape[1]*results.shape[2])
    entropy = results.sum(axis = 1)

    return entropy

def score_kl_divergence(standard_prediction,stochastic_prediction):

    results=np.zeros((stochastic_prediction.shape[0],stochastic_prediction.shape[2],stochastic_prediction.shape[3]))
    for i in range(stochastic_prediction.shape[0]):
        for j in range(stochastic_prediction.shape[2]):
            for k in range(stochastic_prediction.shape[3]):
              results[i,j,k]= kl_div(standard_prediction[i,:,j,k],stochastic_prediction[i,:,j,k]).sum()
    results = results.reshape(results.shape[0], results.shape[1]*results.shape[2])
    kl = results.sum(axis = 1)
    return kl


def score_js_score(standard_prediction,stochastic_prediction):
    
    results=np.zeros((stochastic_prediction.shape[0],stochastic_prediction.shape[2],stochastic_prediction.shape[3]))
    for i in range(stochastic_prediction.shape[0]):
        for j in range(stochastic_prediction.shape[2]):
            for k in range(stochastic_prediction.shape[3]):
              results[i,j,k]= jensenshannon(standard_prediction[i,:,j,k],stochastic_prediction[i,:,j,k]).sum()
    results = results.reshape(results.shape[0], results.shape[1]*results.shape[2])
    js = results.sum(axis = 1)
    return js


