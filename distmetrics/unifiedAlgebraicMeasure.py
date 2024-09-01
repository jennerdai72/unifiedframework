# distmetrics/distmetrics/unifiedAlgebraicMeasure.py

from .functionOnTensor import transform_tensor,transform_2tensors
from .outputsNeuronToProbabilities import outputs_from_neuron_to_probability
from .integralFuncOnProb import integral_funcOnProb_with_options



def unifiedAlgebraicMeasure(source=None, target=None, 
                            metric=None, metricOption=None,
                            sourcePD=None, targetPD=None, probabilityOption=None,
                            sampleFunction=None, sampleFunctionOption=None,
                            probabilityFunctions=None, probabilityFunctionOptions=None,
                            integralFunction=None, integralFunctionOption=None):
    """
    Calculate UAM(Unified Algebraic Measure) that includes UAP(Unified Algebraic Property) or UAD(Unified Algebraic Distance)

    Parameters:
        source, target (str or tensor): Outputs of a neuron or layer from a training dataset,
                                      or the path to a file containing such outputs.
        metric (callable): Function to calculate a measurement of a single output.
        metricOption(dict): metric function option and Output type of a neuron: point,vector,matrix
                            {'MetricType':'p-Norm','p':2} 
        sourcePD, targetPD (tensor): Discrete probability distributions derived from source and target.
        probabilityOption(dict): Options for the probability type.
                                 PMF(Probability Mass Function), 
                                 PDF(Probability Density Function, mainly histogram or binned probabilities), 
                                 CDF (Cumulative Distribution Function), or 
                                 EDF (Empirical Distribution Function).
                                 {'PType':'PMF','Num_bins':100}
        sampleFunction (callable): Sample Function operates source and target sample set.
        sampleFunctionOption (dict): Options for the sample function.
                                  {'SFType':'Euclidean'}{'SFType':'ProbabilityOnEuclidean'}{'SFType':'ProbabilityDistance'}
        probabilityFunctions (callable):[[],[],[]] A list of list of Probability Functions operate on discrete probabilities.
        probabilityFunctionOptions (dict): Options for the probability function.
                                          Correpondingly, each function in [[],[],[]] has a functionOption. 
        integralFunction (callable): Function of integrals over probability function.
        integralFunctionOption (dict): Options for the integral function.

    Returns:
        float: Calculated UAM(UAP or UAD).
    """
    # Handle file inputs for source and target
    if isinstance(source, str):
        with open(source, 'r') as file:
            source = eval(file.read())
    if isinstance(target, str):
        with open(target, 'r') as file:
            target = eval(file.read())

    if source is not None and target is None:
        # Only source is provided, UAP(Unified Algebraic Property)
        # print("target is not provided, begin to calculate UAP(Unified Algebraic Property")
        # print("UAP based on Sample Function")
        # Apply sample function on each item in source
        if sampleFunction is not None:
            source = transform_tensor(source,sampleFunction,sampleFunctionOption)
        
        # Apply metric on each item in source
        if metric is not None:
            source = transform_tensor(source,metric,metricOption) 

        # Convert sample set to probabilities set
        sourcePD = outputs_from_neuron_to_probability(source,probabilityOption)

        #Calculation in Probability Distribution  p_(i) - p_(i-1) Do something, option is put in metricoption
        #metricOption = {'MetricType':'Calcu_In_Dist',‘PConvertFunction’:smoothness}
        if metricOption is not None:
            if metricOption.get('MetricType') == 'Calcu_In_Dist':
                pConvertFunction = metricOption.get('PConvertFunction') 
                sourcePD = pConvertFunction(sourcePD)


        # Apply probability function on sourcePD and targetPD and sum them
        integralOuput = integral_funcOnProb_with_options(probabilityFunctions,probabilityFunctionOptions,sourcePD,targetPD) 

        # Apply integral function to the output
        if integralFunction is not None:
            output = integralFunction(integralOuput, integralFunctionOption)
        # Return UAP
        return output 
    elif source is not None and target is not None:
        # Source and target are provided, UAD(Unified Algebraic Distance)
        # print("source target are provided, begin to calculate UAD(Unified Algebraic Distance)")
        if sampleFunction is not None:
            sample_function_type = sampleFunctionOption.get('SFType', 'Euclidean')
            if sample_function_type == 'Euclidean':
                # print("E_UAD")
                # Apply metric on each item in source
                if metric is not None:
                    source = transform_tensor(source,metric,metricOption)

                output = sampleFunction(source,target,sampleFunctionOption)
                return output
            elif sample_function_type == 'ProbabilityOnEuclidean': #ProbabilityOnEuclidean
                # print("UAD_SF")
                # Apply sample function on each item in source
                if sampleFunction is not None:
                    source = transform_2tensors(source,target,sampleFunction,sampleFunctionOption)
                
                # Apply metric on each item in source
                if metric is not None:
                    source = transform_tensor(source,metric,metricOption) 

                sourcePD = outputs_from_neuron_to_probability(source,probabilityOption)

                integralOuput = integral_funcOnProb_with_options(probabilityFunctions,probabilityFunctionOptions,sourcePD,targetPD) 

                # Apply integral function to the output
                if integralFunction is not None:
                    output = integralFunction(integralOuput, integralFunctionOption)
                    return output
            
            elif sample_function_type == 'ProbabilityDistance':
                # print("UAD_PF")
                # print("source target are provided, begin to calculate UAD_PF(UAD based on Probability Function) ")
                # Apply sample function on each item in source and target
                if sampleFunction is not None:
                    source = transform_tensor(source,sampleFunction,sampleFunctionOption)
                    target = transform_tensor(target,sampleFunction,sampleFunctionOption)
                # Apply metric on each item in source and target
                if metric is not None:
                    source = transform_tensor(source,metric,metricOption)
                    target = transform_tensor(target,metric,metricOption) 
    
                # Convert sample set to probabilities set
                sourcePD = outputs_from_neuron_to_probability(source,probabilityOption)
                targetPD = outputs_from_neuron_to_probability(target,probabilityOption)

                # Apply probability function on sourcePD and targetPD and sum them
                integralOuput = integral_funcOnProb_with_options(probabilityFunctions,probabilityFunctionOptions,sourcePD,targetPD) 

                # Apply function of integral to the output
                if integralFunction is not None:
                    output = integralFunction(integralOuput, integralFunctionOption)
                # Return UAD_PF
                return output 
            else:
                print("Sample function option is not set correctly")
        
    elif source is None and target is None:
        # print("sourcePD targetPD are provided, begin to calculate UAD(Unified Algebraic Distance)")
        if sourcePD is not None and targetPD is None:
            # print("UAP based on Probability Distribution")
            
            #Calculation in Probability Distribution  p_(i) - p_(i-1) Do something, option is put in metricoption
            #metricOption = {'MetricType':'Calcu_In_Dist',‘PConvertFunction’:smoothness}
            if metricOption is not None:
                if metricOption.get('MetricType') == 'Calcu_In_Dist':
                    pConvertFunction = metricOption.get('PConvertFunction') 
                    sourcePD = pConvertFunction(sourcePD)

            # Apply probability function on sourcePD and targetPD and sum them
            integralOuput = integral_funcOnProb_with_options(probabilityFunctions,probabilityFunctionOptions,sourcePD,targetPD) 

            # Apply integral function to the output
            if integralFunction is not None:
                output = integralFunction(integralOuput, integralFunctionOption)
            return output 
        elif sourcePD is not None and targetPD is not None:
            # print("UAD based on Probability Distribution")
            # Apply probability function on sourcePD and targetPD and sum them
            integralOuput = integral_funcOnProb_with_options(probabilityFunctions,probabilityFunctionOptions,sourcePD,targetPD) 

            # Apply function of integral to the output
            if integralFunction is not None:
                output = integralFunction(integralOuput, integralFunctionOption)
            
            return output 
        else:
            print("The arguments are not provided correctly")
    
    