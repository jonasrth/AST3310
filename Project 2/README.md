# Project 2

### Goal
Produce a one-dimensional model of a star, ranging from the surface to the stellar core. This is done by solving a set of differential equations governing the state of the stellar plasma as a function of mass (more stable than solving as a function of radius), starting at the surface and integrating inwards. Energy production is handled by the code from Project 1. 

---

**Contents:**
- **Figures** : Folder holding figures for the project report
- **AST3310___Project_2.pdf** : Written report on the solution of the project.
- **Project2.py** : Python script for solving the project. Includes *StarModel* class that produces a one-dimensional model of a star as a function of a few stellar parameters. The *cross_section* function was provided in the course material.
- **opacity.txt** : Table for opacity as a function of pressure and temperature. Necessary as opacity shows up in the expression for radiative flux.
