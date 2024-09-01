import numpy as np

def tvd_probability_function(probability, probFuncOption=None):
    """
    Calculate the intermediate value for Shannon entropy: P(x_i) * log_b(P(x_i)).

    Parameters:
    probability (float): A single probability value.
    base (int, optional): The logarithm base to use. Default is 2 for bits.

    Returns:
    float: Intermediate value of Shannon entropy calculation.
    """
    
    return np.abs(probability)

def laplacian_probability_function(probability, probFuncOption=None):
    """
    Calculate the intermediate value for laplacian smoothness: P(x_i) * P(x_i).

    Parameters:
    probability (float): A single probability value.
    
    Returns:
    float: Intermediate value of laplacian smoothness calculation.
    """
    return probability*probability

def shannon_probability_function(probability, probFuncOption=None):
    """
    Calculate the intermediate value for Shannon entropy: P(x_i) * log_b(P(x_i)).

    Parameters:
    probability (float): A single probability value.
    base (int, optional): The logarithm base to use. Default is 2 for bits.

    Returns:
    float: Intermediate value of Shannon entropy calculation.
    """
    if probFuncOption is not None:  
        base = probFuncOption.get("base",2)     
    else:
        base = 2
    # Check if the probability is zero to avoid log(0)
    if probability == 0:
        return 0
    
    # Calculate the intermediate value
    intermediate_value = probability * np.log(probability) / np.log(base)
    return intermediate_value

def renyi_probability_function(probability, probFuncOption=None):
    """
    Calculate the intermediate value for Renyi entropy: P(x_i)^alpha.

    Parameters:
    probability (float): A single probability value.
    base (int, optional): The logarithm base to use. Default is 2 for bits.

    Returns:
    float: Intermediate value of Renyi entropy calculation.
    """
    if probFuncOption is not None:  
        base = probFuncOption.get("base",2)
        alpha = probFuncOption.get("alpha",0.5)     
    else:
        base = 2
        alpha = 0.5
    
    # Calculate the intermediate value
    intermediate_value = probability ** alpha
    return intermediate_value

def tsallis_probability_function(probability, probFuncOption=None):
    """
    Calculate the intermediate value for tsallis entropy: P(x_i)^p.

    Parameters:
    probability (float): A single probability value.
    base (int, optional): The logarithm base to use. Default is 2 for bits.

    Returns:
    float: Intermediate value of Tsallis entropy calculation.
    """
    if probFuncOption is not None:  
        q = probFuncOption.get("q",0.5)     
    else:
        q = 0.5
    
    # Calculate the intermediate value
    intermediate_value = probability ** q
    return intermediate_value


def kldivergence_probability_function(probability1,probability2, probFuncOption=None):
    """
    Calculate the intermediate value for KL Divergence: P(x_i) * log(P(x_i)/log(P(y_i))).

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of kl divergence calculation.
    """
    
    # Check if the probability is zero to avoid log(0)
    if probability1 == 0:
        return 0
    if probability2 == 0:
        return 0
     
    # Calculate the intermediate value
    intermediate_value = probability1 * np.log(probability1) / np.log(probability2)
    return intermediate_value

def jsdivergence_probability_function1(probability1,probability2, probFuncOption=None):
    """
    Calculate the intermediate value for js Divergence: P(x_i) * log(P(x_i)/((P(x_i)+P(y_i))/2)).

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of js Divergence calculation.
    """
    
    # Check if the probability is zero to avoid log(0)
    if probability1 == 0:
        return 0
    if probability2 == 0:
        return 0
     
    # Calculate the intermediate value
    intermediate_value = probability1 * np.log(probability1 / ((probability1 + probability2)/2))
    return intermediate_value

def jsdivergence_probability_function2(probability1,probability2, probFuncOption=None):
    """
    Calculate the intermediate value for KL Divergence: Q(x_i) * log(Q(x_i)/((P(x_i)+P(y_i))/2)).

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of js Divergence calculation.
    """
    
    # Check if the probability is zero to avoid log(0)
    if probability1 == 0:
        return 0
    if probability2 == 0:
        return 0
     
    # Calculate the intermediate value
    intermediate_value = probability2 * np.log(probability2 / ((probability1 + probability2)/2))
    return intermediate_value

def alphadivergence_probability_function(probability1,probability2, probFuncOption={'alpha':0.5}):
    """
    Calculate the intermediate value for Alpha Divergence: Q(x_i)^alpha * Q(x_i)^(1-alpha).

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of js Divergence calculation.
    """
    
    if probFuncOption is not None:               
        alpha = probFuncOption.get('alpha') 
     
    # Calculate the intermediate value
    intermediate_value = (probability1**alpha) * (probability2**(1-alpha))
    return intermediate_value

def betadivergence_probability_function1(probability1,probability2, probFuncOption={'beta':0.5}):
    """
    Calculate the intermediate value for Beta Divergence: Q(x_i)^beta.

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of Beta Divergence calculation.
    """
    
    if probFuncOption is not None:               
        beta = probFuncOption.get('beta') 
     
    # Calculate the intermediate value
    intermediate_value = (probability1**beta)
    return intermediate_value

def betadivergence_probability_function2(probability1,probability2, probFuncOption={'beta':0.5}):
    """
    Calculate the intermediate value for Beta Divergence: Q(x_i)^beta.

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of Beta Divergence calculation.
    """
    
    if probFuncOption is not None:               
        beta = probFuncOption.get('beta') 
     
    # Calculate the intermediate value
    intermediate_value = (probability2**beta)
    return intermediate_value

def betadivergence_probability_function3(probability1,probability2, probFuncOption={'beta':0.5}):
    """
    Calculate the intermediate value for Beta Divergence: Q(x_i)^beta.

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of Beta Divergence calculation.
    """
    
    if probFuncOption is not None:               
        beta = probFuncOption.get('beta') 
     
    # Calculate the intermediate value
    intermediate_value = (probability1*probability2**(beta-1))
    return intermediate_value

def gammadivergence_probability_function(probability1,probability2, probFuncOption={'gamma':0.5}):
    """
    Calculate the intermediate value for Gamma Divergence: Q(x_i)^beta.

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of Gamma Divergence calculation.
    """
    
    if probFuncOption is not None:               
        gamma = probFuncOption.get('gamma') 
     
    # Calculate the intermediate value
    intermediate_value = (probability1**gamma)*(probability1**(1-gamma))
    return intermediate_value

def fDefault(x): return x

def fdivergence_probability_function(probability1,probability2, probFuncOption={'f':fDefault}):
    """
    Calculate the intermediate value for f Divergence: Q(x_i)*f(P(x_i)/Q(x_i)).

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of Gamma Divergence calculation.
    """
    
    if probFuncOption is not None:               
        f = probFuncOption.get('f') 
     
    # Calculate the intermediate value
    intermediate_value = (probability2)*f((probability1/probability2))
    return intermediate_value

def Hdivergence_probability_function(probability1,probability2, probFuncOption=None):
    """
    Calculate the intermediate value for H Divergence: (P(x_i)-Q(x_i))^2.

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of H Divergence calculation.
    """
    
    
     
    # Calculate the intermediate value
    intermediate_value = (probability1-probability2)**2
    return intermediate_value

def chi2divergence_probability_function(probability1,probability2, probFuncOption=None):
    """
    Calculate the intermediate value for H Divergence: ((P(x_i)-Q(x_i))^2)/Q(x_i).

    Parameters:
    probability1 (float): A single probability value.
    probability2 (float): A single probability value.
     
    Returns:
    float: Intermediate value of chi2 Divergence calculation.
    """
    
    
     
    # Calculate the intermediate value
    intermediate_value = ((probability1-probability2)**2)/probability2
    return intermediate_value

