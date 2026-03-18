README
Simulation Code for the Autopoietic Matrix Model
(For Editors and Reviewers)
1. General Description
This code implements a computational simulation of the Autopoietic Matrix Model of the Economy presented in the accompanying article.
The model represents the economy as a recursively organized and operationally closed matrix system of products and resources, where production, income generation, demand, savings, and investment form a circular causal structure.
The simulation demonstrates that cyclical dynamics can emerge endogenously from the internal structure of the system, without introducing exogenous shocks.
The code generates time series and graphical representations of:
•	production dynamics
•	consumption dynamics
•	investment in physical capital
•	investment in human capital
•	the endogenous balance trajectory
•	the resulting cyclical dynamics around a rising trend.
The example simulation uses a demonstrative matrix configuration of 20 products and 15 resources.
________________________________________
2. Files Included
The submission package contains:
•	amer_matrix_model_v23_base_working_colab.py
Python simulation code of the model.
•	README file (this document)
Instructions for running the simulation.
•	(optional) sample output files:
o	simulation graphs (PNG)
o	simulation data (CSV)
________________________________________
3. How to Run the Code
The simulation can be executed using Google Colab, which allows running Python code directly in a browser without installing any software.
Step 1
Open Google Colab:
https://colab.research.google.com
Step 2
Create a new notebook.
Step 3
Upload the file:
amer_matrix_model_v23_base_working_colab.py
Step 4
Run the code cell.
The program will automatically:
1.	execute the simulation
2.	generate the time series
3.	display the resulting graphs.
________________________________________
4. Alternative Environments
The code can also be executed in any standard Python environment supporting:
•	Python 3.x
•	NumPy
•	Matplotlib
•	Pandas
Examples:
•	Jupyter Notebook
•	Anaconda
•	VS Code Python environment
________________________________________
5. Reproducibility
All figures presented in the article can be reproduced by executing the provided code.
The simulation uses demonstrative parameter values chosen to illustrate the endogenous cycle mechanism implied by the theoretical model.
The code represents a computational operationalization of the analytical structure described in the article, although some technical details were adapted in order to implement the recursive simulation algorithm.
________________________________________
6. Contact
For questions regarding the model or the simulation code, please contact the author.
