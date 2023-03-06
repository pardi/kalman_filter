![example branch parameter](https://github.com/pardi/kalman_filter/actions/workflows/python-app.yml/badge.svg?branch=main)

# Kalman Filter implementation
This repo contains an implementation of the Kalman Filter. 

To test the algorithm in several conditions, we created a class called SimpleModel that implements a linear system. The noise (mean and std) are option to the class.


## Install

Dependencies:
- python@3.9
- pipenv

The package uses pipenv as virtual environment but a `Makefile` is provided for simplify the pipenv environment.

The `Makefile` calls follows the usage:

`make [ARG]`

with 

``` 
[ARG] 
    install - to setup the environment
    format - to run PEP8 checking and format
    lint - to run the linting on the code
```

## Examples
TODO
