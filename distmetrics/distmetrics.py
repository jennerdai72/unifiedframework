# distmetrics/distmetrics/distmetrics.py
import torch
import numpy as np

#Unified Algebraic Measure Function
from .unifiedAlgebraicMeasure import unifiedAlgebraicMeasure

# import all functions needed
from .sampleFunctionList import *
from .metricFunctionList import *
from .probabilityFunctionList import *
from .functionOfIntegralList import *
from .euclideanDistanceFunctionList import *
from .functionOnTensor import *
#-----------------------------------------------------------
# UAP
#-----------------------------------------------------------

def UAM_shannon_entropy(source,option=None):
    """
    Calculate the Shannon entropy for the given tensor representing the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Shannon entropy.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = None
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[shannon_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'base':option.get("base",2)}]]
    # integralFunction=None
    inFunc = functionOfShannonIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Shannon entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_shannon_entropy_PD(sourcePD,option=None):
    """
    Calculate the Shannon entropy for the given tensor representing the probabilities of the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        tensor (torch.Tensor): A tensor representing the  probabilities of the output of a neuron across all samples.

    Returns:
        float: The Shannon entropy.
    """
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[shannon_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfShannonIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Shannon entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_renyi_entropy(source,option=None):
    """
    Calculate the Renyi entropy for the given tensor representing the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Renyi entropy.
    """
    if option is None:
        option = {}
    else:
        alpha = option.get("alpha",0.5)
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = None
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[renyi_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'base':option.get("base",2),"alpha":alpha}]]
    # integralFunction=None
    inFunc = functionOfRenyiIntegral
    # integralFunctionOption=None 
    inFuncOption = {"alpha":alpha}

    # Calculate Renyi entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_Renyi_entropy_PD(sourcePD,option=None):
    """
    Calculate the Renyi entropy for the given tensor representing the probabilities of the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        tensor (torch.Tensor): A tensor representing the  probabilities of the output of a neuron across all samples.

    Returns:
        float: The Renyi entropy.
    """
    if option is None:
        option = {}
    else:
        alpha = option.get("alpha",0.5)
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[renyi_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'base':option.get("base",2),"alpha":alpha}]]
    # integralFunction=None
    inFunc = functionOfRenyiIntegral
    # integralFunctionOption=None 
    inFuncOption = {"alpha":alpha}

    # Calculate Renyi entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_tsallis_entropy(source,option=None):
    """
    Calculate the Tsallis entropy for the given tensor representing the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        option: {"tsallisP":0.5}

    Returns:
        float: The Tsallis entropy.
    """
    if option is None:
        option = {}
    else:
        tsallisP = option.get("tsallisP",0.5)
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = None
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[tsallis_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'q':tsallisP}]]
    # integralFunction=None
    inFunc = functionOfTsallisIntegral
    # integralFunctionOption=None 
    inFuncOption = {'q':tsallisP}

    # Calculate Tsallis entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_shannon_entropy_PD(sourcePD,option=None):
    """
    Calculate the Shannon entropy for the given tensor representing the probabilities of the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        tensor (torch.Tensor): A tensor representing the  probabilities of the output of a neuron across all samples.
        option: {"tsallisP":0.5}

    Returns:
        float: The Tsallis entropy.
    """
    if option is None:
        option = {}
    else:
        tsallisP = option.get("tsallisP",0.5)
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[tsallis_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'q':tsallisP}]]
    # integralFunction=None
    inFunc = functionOfTsallisIntegral
    # integralFunctionOption=None 
    inFuncOption = {'q':tsallisP}

    # Calculate Shannon entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy



def UAM_TVD_smoothness(source,option=None):
    """
    Calculate the TVD smoothness for the given tensor representing the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The smoothness.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = None
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2),'MetricType':'Calcu_In_Dist','PConvertFunction':TVD_and_Laplacian_Smoothness}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[tvd_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfTVDSmoothnessIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Shannon entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_TVD_smoothness_PD(sourcePD,option=None):
    """
    Calculate the TVD smoothness for the given tensor representing the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The smoothness.
    """
    
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'MetricType':'Calcu_In_Dist','PConvertFunction':TVD_and_Laplacian_Smoothness}
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[tvd_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfTVDSmoothnessIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate TVD using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_Laplacian_smoothness(source,option=None):
    """
    Calculate the Laplacian smoothness for the given tensor representing the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The smoothness.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = None
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2),'MetricType':'Calcu_In_Dist','PConvertFunction':TVD_and_Laplacian_Smoothness}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[laplacian_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfLaplacianSmoothnessIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Laplacian using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy

def UAM_Laplacian_smoothness_PD(sourcePD,option=None):
    """
    Calculate the Laplacian smoothness for the given tensor representing the output of a neuron across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The smoothness.
    """
    
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'MetricType':'Calcu_In_Dist','PConvertFunction':TVD_and_Laplacian_Smoothness}
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {}
    # probabilityFunctions=None
    pFunc = [[laplacian_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfLaplacianSmoothnessIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Laplacian using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    entropy = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return entropy


#-----------------------------------------------------
# UAD_SF on sample function
#-----------------------------------------------------
def UAM_entropy_on_subtraction(source,target,option=None):
    """
    Calculate the Shannon entropy for the given source and target representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Shannon entropy based on the substraction of two tensors.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunction_subtraction
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType",'ProbabilityOnEuclidean')}
    # probabilityFunctions=None
    pFunc = [[shannon_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'base':option.get("base",2)}]]
    # integralFunction=None
    inFunc = functionOfShannonIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Shannon entropy using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance
#--------------------------------------
# E_UAD
#--------------------------------------
def UAM_MMD(source,target,option=None):
    """
    Calculate the MMD(Maximum Mean Discrepency) for the given source and target representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The MMD(Maximum Mean Discrepency).
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = compute_mmd
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType",'Euclidean'),'kernel':option.get("kernel",rbf_kernel),'gamma':option.get("gamma",1.0) }
    # probabilityFunctions=None
    pFunc = None
    # probabilityFunctionOptions=None
    pFuncOption = None
    # integralFunction=None
    inFunc = None
    # integralFunctionOption=None 
    inFuncOption = None

    # Calculate MMD using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance

#-----------------------------------------------------------
# UAD_PF
#-----------------------------------------------------------

def UAM_kl_divergence(source,target,option=None):
    """
    Calculate the KL_Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The KL Divergence.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[kldivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfklDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate KL Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_kl_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the KL_Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The KL Divergence.
    """
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[kldivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfklDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate KL Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_js_divergence(source,target,option=None):
    """
    Calculate the Jensen Shannon_Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Jensen Shannon Divergence.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[jsdivergence_probability_function1,jsdivergence_probability_function2]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{},{}]]
    # integralFunction=None
    inFunc = functionOfjsDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate JS Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_js_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the Jensen Shannon_Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The Jensen Shannon Divergence.
    """
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[jsdivergence_probability_function1,jsdivergence_probability_function2]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{None},{None}]]
    # integralFunction=None
    inFunc = functionOfjsDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate JS Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance

def UAM_alpha_divergence(source,target,option=None):
    """
    Calculate the Alpha_Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Alpha Divergence.
    """
    if option is None:
        option = {}
    else:
        alpha = option.get('alpha')

    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[alphadivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'alpha':alpha}]]
    # integralFunction=None
    inFunc = functionOfalphaDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {'alpha':alpha}

    # Calculate Alpha Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_alpha_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the Alpha_Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The Alpha Divergence.
    """
    if option is None:
        option = {}
    else:
        alpha = option.get('alpha')
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[alphadivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'alpha':alpha}]]
    # integralFunction=None
    inFunc = functionOfalphaDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {'alpha':alpha}

    # Calculate Alpha Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance

def UAM_beta_divergence(source,target,option=None):
    """
    Calculate the Beta Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Beta Divergence.
    """
    if option is None:
        option = {}
    else:
        beta = option.get('beta')
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[betadivergence_probability_function1,betadivergence_probability_function2,betadivergence_probability_function3]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'beta':beta},{'beta':beta},{'beta':beta}]]
    # integralFunction=None
    inFunc = functionOfbetaDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {'beta':beta}

    # Calculate Beta Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_beta_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the Beta Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The Beta Divergence.
    """
    if option is None:
        option = {}
    else:
        beta = option.get('beta')
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[betadivergence_probability_function1,betadivergence_probability_function2,betadivergence_probability_function3]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{'beta':beta},{'beta':beta},{'beta':beta}]]
    # integralFunction=None
    inFunc = functionOfbetaDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {'beta':beta}

    # Calculate Beta Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance

def UAM_gamma_divergence(source,target,option=None):
    """
    Calculate the Gamma Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Gamma Divergence.
    """
    if option is None:
        option = {}
    else:
        gamma = option.get('gamma')
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[gammadivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{"gamma":gamma}]]
    # integralFunction=None
    inFunc = functionOfgammaDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {"gamma":gamma}

    # Calculate Gamma Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_gamma_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the Gamma Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The Gamma Divergence.
    """
    if option is None:
        option = {}
    else:
        gamma = option.get('gamma')
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[gammadivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{"gamma":gamma}]]
    # integralFunction=None
    inFunc = functionOfgammaDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {"gamma":gamma}

    # Calculate Gamma Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance

def UAM_f_divergence(source,target,option=None):
    """
    Calculate the f Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The f Divergence.
    """
    def f(x): return x
    if option is None:
        option = {}
    else:
        f = option.get("f",f)
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[fdivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{"f":f}]]
    # integralFunction=None
    inFunc = functionOffDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate f Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_f_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the f Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The f Divergence.
    """
    def f(x): return x
    if option is None:
        option = {}
    else:
        f = option.get("f",f)
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[fdivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{"f":f}]]
    # integralFunction=None
    inFunc = functionOffDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate f Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance

def UAM_H_divergence(source,target,option=None):
    """
    Calculate the H Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The H Divergence.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[Hdivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfHDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate H Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_H_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the H Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The H Divergence.
    """
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[Hdivergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfHDivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate H Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance

def UAM_chi2_divergence(source,target,option=None):
    """
    Calculate the Chi^2 Divergence for the given two tensors representing the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        source (torch.Tensor): A tensor representing the output of a neuron across all samples.
        target (torch.Tensor): A tensor representing the output of a neuron across all samples.

    Returns:
        float: The Chi^2 Divergence.
    """
    if option is None:
        option = {}
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = source
    # target=None
    t = target
    # metric=None
    mFunc = p_norm_metric
    # metricOption=None
    mOption = {'p':option.get("p",2)}
    # sourcePD=None
    sPD = None
    # targetPD=None
    tPD = None
    # probabilityOption=None
    pOption = {'PType':option.get("PType","PDF"),'Num_bins':option.get("Num_bins",100)}
    # sampleFunction=None
    sFunc = sampleFunctionNoChange
    # sampleFunctionOption=None
    sFuncOption = {'SFType':option.get("SFType","ProbabilityDistance")}
    # probabilityFunctions=None
    pFunc = [[chi2divergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfchi2DivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Chi^2 Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance


def UAM_Chi2_divergence_PD(sourcePD,targetPD,option=None):
    """
    Calculate the Chi^2 Divergence for the given two tensors representing the probabilities of the outputs of two neurons across all samples.
    Invokes the unifiedAlgebraicMeasure function.

    Args:
        sourcePD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.
        targetPD (torch.Tensor): A tensor representing the probabilities of the output of a neuron across all samples.

    Returns:
        float: The Chi^2 Divergence.
    """
    # Prepare arguements for unifiedAlgebraicMeasure
    # source=None
    s = None
    # target=None
    t = None
    # metric=None
    mFunc = None
    # metricOption=None
    mOption = None
    # sourcePD=None
    sPD = sourcePD
    # targetPD=None
    tPD = targetPD
    # probabilityOption=None
    pOption = None
    # sampleFunction=None
    sFunc = None
    # sampleFunctionOption=None
    sFuncOption = None
    # probabilityFunctions=None
    pFunc = [[chi2divergence_probability_function]]
    # probabilityFunctionOptions=None
    pFuncOption = [[{}]]
    # integralFunction=None
    inFunc = functionOfchi2DivergenceIntegral
    # integralFunctionOption=None 
    inFuncOption = {}

    # Calculate Chi^2 Divergence using unifiedAlgebraicMeasure
    # Modify the arguments to unifiedAlgebraicMeasure as needed
    distance = unifiedAlgebraicMeasure(source=s, target=t, 
                            metric=mFunc, metricOption=mOption,
                            sourcePD=sPD, targetPD=tPD, probabilityOption=pOption,
                            sampleFunction=sFunc, sampleFunctionOption=sFuncOption,
                            probabilityFunctions=pFunc, probabilityFunctionOptions=pFuncOption,
                            integralFunction=inFunc, integralFunctionOption=inFuncOption )

    return distance








# Example usage
sourceTest = torch.tensor([0.1, 0.2, 0.3, 0.4])
targetTest = torch.tensor([0.2, 0.4, 0.6, 0.8])
print(sourceTest.dim())
print("Begin to calculate Shannon Entropy")
entropy = UAM_shannon_entropy(sourceTest)
print(f"Shannon Entropy: {entropy}")
entropy = UAM_shannon_entropy_PD(sourceTest)
print(f"Shannon Entropy on subtraction: {entropy}")
entropy = UAM_entropy_on_subtraction(sourceTest,targetTest)
print(f"Shannon Entropy on subtraction: {entropy}")
mmd = UAM_MMD(sourceTest,targetTest)
print(f"MMD: {mmd}")
kl = UAM_kl_divergence(sourceTest,targetTest)
print(f"KL Divergence: {kl}")
kl = UAM_kl_divergence_PD(sourceTest,targetTest)
print(f"KL Divergence: {kl}")
js = UAM_js_divergence(sourceTest,targetTest)
print(f"JS Divergence: {js}")
js = UAM_js_divergence_PD(sourceTest,targetTest)
print(f"JS Divergence: {js}")

smoothness = UAM_TVD_smoothness(sourceTest)
print(f"TVD Smoothness From Sample: {smoothness}")
smoothness = UAM_TVD_smoothness_PD(sourceTest)
print(f"TVD Smoothness From Probability: {smoothness}")
smoothness = UAM_Laplacian_smoothness(sourceTest)
print(f"Laplacian Smoothness From Sample: {smoothness}")
smoothness = UAM_Laplacian_smoothness_PD(sourceTest)
print(f"Laplacian Smoothness From Probability: {smoothness}")
