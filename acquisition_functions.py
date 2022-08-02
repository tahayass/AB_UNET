from scipy.stats import entropy as ent
import numpy as np
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
import timeit





def score_entropy(stochastic_prediction):

    results=np.zeros((stochastic_prediction.shape[0],stochastic_prediction.shape[2],stochastic_prediction.shape[3]))
    for i in range(stochastic_prediction.shape[0]):
        for j in range(stochastic_prediction.shape[2]):
            for k in range(stochastic_prediction.shape[3]):
                results[i,j,k]= ent(stochastic_prediction[i,:,j,k])
    entropy=results.sum()

    return entropy





def main():
    a=np.random.rand(1,4,300,300)
    print(score_entropy(a))
    


if __name__ == "__main__":
    main()
