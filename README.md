# CBF-CLF-Helper
Matlab library for Control Barrier Function (CBF) and Control Lyapunov Function (CLF) based control methods. The library is designed to let users easily implement safety controller based on CBFs and CLFs with Matlab. We provide:
- An easy interface for construction and simulation of nonlinear control-affine systems.
- Safety controllers including CLF-QP, CBF-QP, and CBF-CLF-QP as built-in functions.
- Demonstrations on toy examples.


## 1. Requirements
- Matlab
- [Symbolic Math Toolbox](https://www.mathworks.com/products/symbolic.html)
- [Optimization Toolbox](https://www.mathworks.com/products/optimization.html)



## 2. Usage
1. Create a class that inherit `CtrlAffineSys`.
3. Create class functions `defineClf` and `defineCbf` and define your CLF and CBF in each function respectively using the same symbolic expressions.
2. Create a class function `defineSystem` and define your dynamics using the symbolic toolbox.
4. To run the simulation or run the controller, create a class instance with parameters specified as a Matlab structure array, and use the built-in functions—dynamics and other controllers such as `ctrlCbfClfQp`, `ctrlClfQp`, etc.

Please checkout the [Manual](https://github.com/HybridRobotics/CBF-CLF-Helper/blob/master/Manual_v1.pdf) for more details.

### Running over SSH Server
Checkout [note_on_ssh.md](./note_on_ssh.md) for running the code over a matlab SSH server.

## 3. Demos
Run files in the directory `demos` in Matlab.

## 4. Citation
```
@misc{choi2020cbfclfhelper,
  author       = {Jason J. Choi},
  title        = {CBF-CLF-Helper 1.0: Library for Control Barrier Function (CBF) and Control Lyapunov Function (CLF) based control methods},
  year         = {2020},
  version      = {1.0.0},
  url          = {https://github.com/HybridRobotics/CBF-CLF-Helper}
}
```

## 5. Notes
Unofficial 2.0 version is under development at [this repo](https://github.com/ChoiJangho/CBF-CLF-Helper).
If you want to see new demos, please check out this [folder](https://github.com/ChoiJangho/CBF-CLF-Helper/tree/feedback_linearization/demos).
