# PokeChallenge 2026 final refactoring

## File tree
- domain/
  Domain folder contains all the files needed to implement the logic for Pokemon gen 2 battle simulations.
  It also contains interfaces that can be used to use custom logic for handling data artifacts and errors. 
  
  For example:
  At the moment we are using a knockout detector which is fixed and hard to maintain and test, i want to implement a common interfaces for adding such detectors 

  Domain should be an independent package that can be used in code, but also used from cli to do testing ecc...
  Build pokedex from train json file ecc
  IT MUST BE TESTABLE IDEPENDENTLY
  
  - addons/
    one directory for each of the external logic such as knockout models ecc.

- assets/
  only contains the .csv for damage multipliers supplied by the competition

- features/ 
  - core/
    contains the concrete implementation of each feature
  - /
    contains one file for feature_set

- models/
  - core/
    the base implementation of model wrappers
  - /
    contains one file for concrete model (currently model catalog)

- / 
  contains interfaces implementation for models, feature, feature_set ecc.
  contains the gym implementation and the main file for running evaluations, training, tuning and testing.

## Architecture needs
