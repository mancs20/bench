# Guide to use MiniZinc models and data files in the multi-objective experiments

## Choco-solver and MiniZinc

If we have a problem modeled in MiniZinc, we do not need to remodeled again in choco. Instead, we can flatten the model
with the data into a flatzinc file (.fzn) and that will be the instance that we should pass to choco. NOTE: The flatzinc
should have extension .fzn.

## Steps before flattening

1. All the objectives should be saved in an array called `objs`, something like this:
```minizinc     
array[1..3] of var int: objs;
constraint objs[1] = total_cost;
constraint objs[2] = max_resolution;
constraint objs[3] = max_incidence;
```
2. All the objectives should be minimized or maximized, depending on the problem. We can convert a minimization problem
to a maximization problem by multiplying the objective by -1. We have to indicate in the model if the objectives are 
minimized or maximized. We should add the dummy variable ```minimization```  or ```maximization``` to the model. 
For example for a minimization problem:
```minizinc
var bool: minimization;
```
3. In case we are interested not only on the values of the objectives but also on the values of the decision variables,
we should group all the decision variables in an array or set called `decisions_vars`. In the example below, the 
decision variables are the images that you select to be part of the solution. The name should be all in upper case or all in lower case.
```minizinc
array[IMAGES] of var bool: decision_vars; % take the image i or not
```
4. IMPORTANT. To calculate the hypervolume indicator, we need to know the reference point. The reference point is the 
maximum or minimum value of each objective, so bound properly the objectives.

## Flattening the model and the data

To flatten the model and the data, we can use the MiniZinc command line from the terminal. The command to flatten the model and the data is:
```bash
minizinc -c --solver org.minizinc.mzn-fzn -I "path/to/choco/mzn_lib" model_to_flatten.mzn -d data_for_model.dzn -o output_flatzinc.fzn
```

### Possible issues
At the moment of writing this document, the flattening process produces a compatibility issues with choco solver, check 
https://github.com/chocoteam/choco-solver/issues/1074. Choco 4.10.14 and MiniZinc 2.8.5.
This can be solved by adding the following line to the choco file, `parsers/src/main/minizinc/mzn_lib/redefinitions-2.5.2.mzn`:
```java
predicate array_var_float_element2d_nonshifted(var int: idx1, var int: idx2, array[int,int] of var float: x, var float: c) =
  let {
    int: dim = card(index_set_2of2(x));
    int: min_flat = min(index_set_1of2(x))*dim+min(index_set_2of2(x))-1;
  } in array_var_float_element_nonshifted((idx1*dim+idx2-min_flat)::domain, array1d(x), c);

predicate array_var_set_element2d_nonshifted(var int: idx1, var int: idx2, array[int,int] of var set of int: x, var set of int: c) =
  let {
    int: dim = card(index_set_2of2(x));
    int: min_flat = min(index_set_1of2(x))*dim+min(index_set_2of2(x))-1;
  } in array_var_set_element_nonshifted((idx1*dim+idx2-min_flat)::domain, array1d(x), c);
```

## Using MiniZinc from the command-line
To use MiniZinc from the command-line you will have to add the installation location, where the minizinc executables are,
to the your ```PATH```. For in bash and related shells this can be done using export 
```PATH=$PATH:{MINIZINC}``` where ```{MINIZINC}``` is the installation location. 
(Note that this is for the current session, if you close the terminal you will have to do it again). 
Taken from https://github.com/MiniZinc/libminizinc/issues/213.

### MacOS

If you have installed MiniZinc in the global Applications folder the installation location should be
```/Applications/MiniZincIDE.app/Contents/Resources/```. Otherwise if you have installed it in the user applications 
folder than it should be ```~/Applications/MiniZincIDE.app/Contents/Resources/```.

A simple check can show you the correct answer:
```bash
ls /Applications/MiniZincIDE.app/Contents/Resources/
```
should output something like
```bash
fzn-chuffed     fzn-gecode-gist minizinc        mzn-chuffed     mzn-g12lazy     mzn-gecode      mzn2doc         mznide.icns     share           solns2out
flatzinc        fzn-gecode      gecode          mzn-cbc         mzn-g12fd       mzn-g12mip      mzn-gurobi      mzn2fzn         qt.conf         solns2dzn
```

#### Add minizinc permanently to PATH

To add it permanently you can edit the file ``~/.bash_profile``, and add this:
```bash
# Minizinc
PATH=${PATH}:/Applications/MiniZincIDE.app/Contents/Resources
```
Check https://scriptingosx.com/2017/04/about-bash_profile-and-bashrc-on-macos/
and https://scriptingosx.com/2017/04/on-bash-environment-variables/. 

