# Manual installation

## Requirements 
[Git](https://git-scm.com/downloads)
[Python 3.5](https://www.python.org/downloads/release/python-353/)

## Virtual environments (optional)

It's highly recommended to work with virtualenv if you install the project manually. It's easier to manage the installed packages with it.

To install it, run the following command in a terminal :

```
pip install virtualenv virtualenvwrapper-win
```

Then create a virtualenv :

```
mkvirtualenv [envName]
```

You can then start working in the environment. To exit it, use the following command :

```
deactivate
```

When you want to start the environment again use :

```
workon [envName]
```

For more information you can go on the [official documentation](https://virtualenvwrapper.readthedocs.io/en/latest/)

## pip

Numpy and scipy are not well supported on Windows. You have to download them ([Numpy link](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) and [Scipy link](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)). Take the version cp35 and amd64 version and put them in this folder.

To install them (modify the filenames if necessary) :

```
pip install numpy‑1.13.1+mkl‑cp35‑cp35m‑win_amd64.whl
pip install scipy‑0.19.1‑cp35‑cp35m‑win_amd64.whl
```

Then you can install the rest with :

```
pip install -r requirements-cpu.txt
```

GPU version : 
```
pip install -r requirements-gpu.txt
```

## Running the code

Once everything is installed, you can launch jupyter :

```
jupyter notebook [pathToWorkspace]
```