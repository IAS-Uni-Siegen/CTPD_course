## How to use the notebooks:

### Installing python on Windows:


- install Python 3.13 from https://www.python.org/
    - e.g.: https://www.python.org/downloads/release/python-3144/
    - the default installation path in Windows is ```C:\Users\{user_name}\AppData\Local\Programs\Python\Python314```
- setup a virtual environment with ```{path/to/python/exe} -m venv {env-name}```
    - e.g. ```C:\Users\hvater\AppData\Local\Programs\Python\Python314\python -m venv C:\Users\hvater\work\venvs\ctpd-venv```
- activate the venv with ```.{path/to/venv/scripts}```
    - e.g. ```.\work\venvs\ctpd-venv\Scripts\activate```
- use ```pip install requirements.txt``` in the ```interactive_notebooks```-directory to install all required python packages
- run ```jupyter-lab``` within the activated venv
    - alternatively, you may use ```jupyter-notebook``` or the notebook functionalities from within, e.g., vscode