import math

def functionOfTVDSmoothnessIntegral(tvdSmoothnessIntegral,Option=None):
    """
    Return TVD smoothness.

    Args:
        tvdSmoothnessIntegral: The integral of funtion of probability.

    Returns:
        TVD smoothness.
    """
    return tvdSmoothnessIntegral[0][0]

def functionOfLaplacianSmoothnessIntegral(laplacianSmoothnessIntegral,Option=None):
    """
    Return laplacian smoothness.

    Args:
        laplacianSmoothnessIntegral: The integral of funtion of probability.

    Returns:
        laplacian smoothness.
    """
    return laplacianSmoothnessIntegral[0][0]

def functionOfShannonIntegral(shannonIntegral,Option=None):
    """
    Return Shannon Entropy.

    Args:
        shannonIntegral: The integral of funtion of probability.

    Returns:
        Shannon Entropy.
    """
    return -shannonIntegral[0][0]

def functionOfRenyiIntegral(renyiIntegral,Option=None):
    """
    Return Renyi Entropy.

    Args:
        renyiIntegral: The integral of funtion of probability.

    Returns:
        Renyi Entropy.
    """
    if Option is not None:  
        base = Option.get("base",2)
        alpha = Option.get("alpha",0.5)     
    else:
        base = 2
        alpha = 0.5
    return math.log(renyiIntegral[0][0], base)/(1-alpha)

def functionOfTsallisIntegral(tsallisIntegral,Option=None):
    """
    Return Renyi Entropy.

    Args:
        renyiIntegral: The integral of funtion of probability.

    Returns:
        Renyi Entropy.
    """
    if Option is not None:  
        q = Option.get("q",0.5)     
    else:
        q = 0.5
    return (1 - tsallisIntegral[0][0])/(q - 1)


def functionOfklDivergenceIntegral(klDivergenceIntegral,Option=None):
    """
    Return KL Divergence.

    Args:
        klDivergenceIntegral: The integral of funtion of probability.

    Returns:
        KL Divergence.
    """
    return klDivergenceIntegral[0][0]

def functionOfjsDivergenceIntegral(jsDivergenceIntegral,Option=None):
    """
    Return js Divergence.

    Args:
        jsDivergenceIntegral: The integral of funtion of probability.

    Returns:
        js Divergence.
    """
    
    return (jsDivergenceIntegral[0][0]+jsDivergenceIntegral[0][1])/2

def functionOfalphaDivergenceIntegral(alphaDivergenceIntegral,Option={'alpha':0.5}):
    """
    Return alpha Divergence.

    Args:
        alphaDivergenceIntegral: The integral of funtion of probability.

    Returns:
        alpha Divergence.
    """
    if Option is not None:               
        alpha = Option.get('alpha') 
    return (1 - alphaDivergenceIntegral[0][0])/(alpha*(alpha -1))

def functionOfbetaDivergenceIntegral(betaDivergenceIntegral,Option={'beta':0.5}):
    """
    Return beta Divergence.

    Args:
        betaDivergenceIntegral: The integral of funtion of probability.

    Returns:
        beta Divergence.
    """
    if Option is not None:               
        beta = Option.get('beta') 
    return (betaDivergenceIntegral[0][0]-beta*betaDivergenceIntegral[0][1]-(beta - 1)*betaDivergenceIntegral[0][2])/(beta*(beta -1))

def functionOfgammaDivergenceIntegral(gammaDivergenceIntegral,Option={'gamma':0.5}):
    """
    Return gamma Divergence.

    Args:
        gammaDivergenceIntegral: The integral of funtion of probability.

    Returns:
        gamma Divergence.
    """
    if Option is not None:               
        gamma = Option.get('gamma') 
    return (gammaDivergenceIntegral[0][0]-1)/(gamma*(gamma -1))

def functionOffDivergenceIntegral(fDivergenceIntegral,Option=None):
    """
    Return f Divergence.

    Args:
        fDivergenceIntegral: The integral of funtion of probability.

    Returns:
        f Divergence.
    """
    
    return fDivergenceIntegral[0][0]

def functionOfHDivergenceIntegral(HDivergenceIntegral,Option=None):
    """
    Return H Divergence.

    Args:
        HDivergenceIntegral: The integral of funtion of probability.

    Returns:
        H Divergence.
    """
    
    return (HDivergenceIntegral[0][0]/2)**(1/2)

def functionOfchi2DivergenceIntegral(chi2DivergenceIntegral,Option=None):
    """
    Return Chi2 Divergence.

    Args:
        chi2DivergenceIntegral: The integral of funtion of probability.

    Returns:
        Chi2 Divergence.
    """
    
    return chi2DivergenceIntegral[0][0]
