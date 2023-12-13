# Model Based Reinforcement Learning for Learning Control of A Double Integrator

Repo that generates code used in the Final Project for the class CS 5180 Reinforcement Learning at Northeastern University, Boston.

Objective:
Generate a time optimal speed profile for any non-linear dynamics resulting in a non-linear controller without any expert knowledge using a Model Based Reinforcement Learning algorithm called Probabilistic Inference for Learning Control (PILCO) This control policy is time optimal in generating inputs that produce rest-to-rest motion. 

# Main code 
* Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control\examples\double_integrator\main.ipynb
* cvxpy.ipynb

# Usage

    git clone https://github.com/rsankhye/CS5180-PILCO.git
    cd CS5180-PILCO
    conda env create -f environment.yml
    conda activate CS5180-PILCO
    conda develop .
