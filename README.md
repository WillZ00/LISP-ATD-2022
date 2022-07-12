###Installation
To create a conda environment containing the necessary software to run the example forecasters, you will need to run one of the following two commands, depending on your system's platform (operating system and CPU instruction set).
Note: You may need to toggle which line is commented between lines 43/44 in environment.yml (or lines 42/43 in environment_osx-arm64.yml), depending on whether you authenticate to Gitlab using ssh or https.
If you are running an x64 processor in Windows, Linux, or OSX, run:

mamba env create -f environment.yml  # If win-64, linux-64, or osx-64


Otherwise, if you are running an ARM64-based M1 chip on OSX, run:

mamba env create -f environment_osx-arm64.yml  # If M1-based Mac


After creating your environment, you will need to activate your environment for your current terminal. You can do this by running,

mamba activate LISP-ATD-2022


Finally, you will need to create a Jupyter kernel to use your environment within Jupyter notebooks. To do this, run,

python -m ipykernel install --name LISP-ATD-2022 --user


###Below is an overview of the purposes of each file in this repo:

my_mod.py -- our customized python module that supports our prediction pipeline. (The pre-configed model is located at the bottom of the file)

1D_CNN_20Dim_Full_Pred.ipynb -- running all cells in this notebook will generate all four timesteps of predictions for all the regions. 
    The prediction results is stored in pandas.DataFrame format. The dataframe will also be saved locally as a CSV file.

1D_CNN_20Dim.ipynb -- This jupyter notebook can be ignored. We used it for developing and testing our models.

EVN.yml -- 

setup.cfg --
