"""Script used to let JupyterLab notebook import other notebooks as modules. It also
contains a definition that lets import all necesary pacjakes and modules for the repository."""

def nbimporter(activate=True):
    if activate == True:
        import io, os, sys, types
        from IPython import get_ipython
        from nbformat import read
        from IPython.core.interactiveshell import InteractiveShell

        def find_notebook(fullname, path=None):
            """find a notebook, given its fully qualified name and an optional path

            This turns "foo.bar" into "foo/bar.ipynb"
            and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
            does not exist.
            """
            name = fullname.rsplit('.', 1)[-1]
            if not path:
                path = ['']
            for d in path:
                nb_path = os.path.join(d, name + ".ipynb")
                if os.path.isfile(nb_path):
                    return nb_path
                # let import Notebook_Name find "Notebook Name.ipynb"
                nb_path = nb_path.replace("_", " ")
                if os.path.isfile(nb_path):
                    return nb_path

        class NotebookLoader(object):
            """Module Loader for Jupyter Notebooks"""

            def __init__(self, path=None):
                self.shell = InteractiveShell.instance()
                self.path = path

            def load_module(self, fullname):
                """import a notebook as a module"""
                path = find_notebook(fullname, self.path)

                print("importing Jupyter notebook from %s" % path)

                # load the notebook object
                with io.open(path, 'r', encoding='utf-8') as f:
                    nb = read(f, 4)

                # create the module and add it to sys.modules
                # if name in sys.modules:
                #    return sys.modules[name]
                mod = types.ModuleType(fullname)
                mod.__file__ = path
                mod.__loader__ = self
                mod.__dict__['get_ipython'] = get_ipython
                sys.modules[fullname] = mod

                # extra work to ensure that magics that would affect the user_ns
                # actually affect the notebook module's ns
                save_user_ns = self.shell.user_ns
                self.shell.user_ns = mod.__dict__

                try:
                    for cell in nb.cells:
                        if cell.cell_type == 'code':
                            # transform the input to executable Python
                            code = self.shell.input_transformer_manager.transform_cell(cell.source)
                            # run the code in themodule
                            exec(code, mod.__dict__)
                finally:
                    self.shell.user_ns = save_user_ns
                return mod

        class NotebookFinder(object):
            """Module finder that locates Jupyter Notebooks"""

            def __init__(self):
                self.loaders = {}

            def find_module(self, fullname, path=None):
                nb_path = find_notebook(fullname, path)
                if not nb_path:
                    return

                key = path
                if path:
                    # lists aren't hashable
                    key = os.path.sep.join(path)

                if key not in self.loaders:
                    self.loaders[key] = NotebookLoader(path)
                return self.loaders[key]
                
        # NotebookFinder class is added to metapath to be able to import *.ipynb files.
        sys.meta_path.append(NotebookFinder())
    else:
        pass
    
def pmimporter(activate=True):
    """Definition that imports all packages if boolean variable activate, is True"""
    if activate == True:
        import numpy as np
        from pathlib import Path
        from tqdm import tqdm
        import pandas as pd
        import pickle
        from joblib import load
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from matplotlib import pyplot as plt
        import pandas as pd
    else:
        pass

def modfactors(X1,X2,X5,X6,X7,X10,X13):
    """Definition to execute the estimation of modification factors."""
    # Directories
    directory = Path.cwd()
    datadir = directory / 'Data'

    # Figure transformation factor
    cm = 1/2.54

    # Loading predictors in scaled-standardized and unscaled-unstandardized conditions
    with open(datadir/'PsiDataStats.pickle','rb') as f:
        DCRac,Psi,Xi,Xi_NS,_,_,_,_,_,_,_,_,_,_,_,_ = pickle.load(f)
    
    # Creating Pandas Series
    Parameters = [X1,X2,X5,X6,X7,X10,X13]
    PredList = ['X1','X2','X5','X6','X7','X10','X13']            # Predictors to be used
    PredData = pd.Series(data=Parameters,index=PredList)
    
    # Calculation of Alpha values
    '''Using classical regression equations'''
    alpha10th = 0.281*PredData.loc['X6'] + 0.368
    alpha50th = 0.361*PredData.loc['X6'] + 0.339
    alpha90th = 0.386*PredData.loc['X6'] + 0.429
    '''Using ML-GBRT model'''
    GBRTFilePath = datadir/'alpha_regressionModel.joblib'
    RegModel,TModel,trainedParams = load(GBRTFilePath)
    
    '''Predictors from PredList are located in their indexes'''
    PreData4alpha = Xi['MRSA']['convDs']['fixBase']['1N'].loc[:,['X1','X5','X6','X7']]
    # Standardization
    PredictorStandScaler = StandardScaler().fit(PreData4alpha.values)
    StandScaledPredictors = PredictorStandScaler.transform(PreData4alpha.values)
    # MinMax Scaling
    PredictorScaler = MinMaxScaler(feature_range=(0,1)).fit(StandScaledPredictors)
    # Scaled and standardized values
    SN_PredData4Alpha = PredictorScaler.transform(PredictorStandScaler.transform(PredData.loc[['X1','X5','X6','X7']].values.reshape(1,-1)))

    '''alpha prediction with GBRT model'''
    alphaGBRT = RegModel.predict(SN_PredData4Alpha)[0]
    
    
    # ----------------------------------------------------------------------
    # AlphaCd values
    # ----------------------------------------------------------------------
    alphaCd10thX5 = 1/(0.06*PredData.loc['X5'] + 0.539)
    alphaCd50thX5 = 1/(0.069*PredData.loc['X5'] + 0.164)
    alphaCd90thX5 = 1/(0.056*PredData.loc['X5'] + 0.134)

    '''Using equations with X10'''
    alphaCd10thX10 = 1.032*PredData.loc['X10']**-1.11 - 0.3874
    alphaCd50thX10 = -2.683*PredData.loc['X10'] + 3.458
    alphaCd90thX10 = -1.618*PredData.loc['X10']**2.483 + 2.67

    '''Using GBRT model'''
    with open(datadir/'GBRTModelCd.pickle','rb') as f:
        reg,TrainedModel = pickle.load(f)

    '''Predictors from PredList are located in their indexes and the rest are replaced by its mean values'''
    InputDataList = ['X1','X2','X5','X10','X13']
    # PreData4alphaCd = Xi['MRSA']['convDs']['fixBase']['1N'].loc[:,['X1','X2','X5','X10','X13']]
    CompletePredList = ['X1','X2','X3','X4','X5','X10','X12','X13']
    '''Array with data for the evaluated structure'''
    DataList4alphaCd = np.zeros(len(CompletePredList))
    for i in range(len(CompletePredList)):
        if CompletePredList[i] in InputDataList:
            DataList4alphaCd[i] = PredData.loc[CompletePredList[i]]
        else:
            DataList4alphaCd[i] = np.mean(Xi['MRSA']['convDs']['fixBase']['1N'].loc[:,CompletePredList[i]].values)

    # Standardization
    PreData4alphaCd = Xi['MRSA']['convDs']['fixBase']['1N'].loc[:,CompletePredList]
    PredictorStandScaler = StandardScaler().fit(PreData4alphaCd.values)
    StandScaledPredictors = PredictorStandScaler.transform(PreData4alphaCd.values)
    # MinMax Scaling
    PredictorScaler = MinMaxScaler(feature_range=(0,1)).fit(StandScaledPredictors)
    # Scaled and standardized values
    SN_PredData4AlphaCd = PredictorScaler.transform(PredictorStandScaler.transform(DataList4alphaCd.reshape(1,-1)))

    '''alpha prediction with GBRT model'''
    alphaCdGBRT = reg.predict(SN_PredData4AlphaCd)[0]

    # ---------------------------------------------------------------------------------------------------------------------
    # Consolidating data in a DataFrame
    # ---------------------------------------------------------------------------------------------------------------------
    alphaVals = np.array([alpha10th,alpha50th,alpha90th,alphaGBRT])
    alphaCdVals = np.array([alphaCd10thX5,alphaCd50thX5,alphaCd90thX5,alphaCd10thX10,alphaCd50thX10,alphaCd90thX10,alphaCdGBRT])
    
    return alphaVals, alphaCdVals