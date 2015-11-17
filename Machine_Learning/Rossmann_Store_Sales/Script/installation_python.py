#!/usr/bin/python

# Daniel Dixey
# Installing the Python Modules required to run the Script

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', '--upgrade', package])
        
if __name__ == "__main__":
    # List of Packages
    req_packages = ['matplotlib',
                    'pandas',
                    'seaborn',
                    'scikit-learn',
                    'numpy',
                    'scipy',
                    'thinc']
    # Iterate through each package and check its installed
    for each_item in req_packages:
        install_and_import(each_item)
