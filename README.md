# Modelling and Analysis of a Vehicle Routing Problem with Time Windows in Freight Delivery

## Introduction
A MSc's Dissertation Project which focuses on Vehicle Routing Problem with Time Windows (VRPTW), using both exact method and heuristic approach (General Variable Neighbourhood Search - GVNS).

The exact approach is based on a classical Mix-Integer Programming (MIP) model and solved by IBM ILOG CPLEX 12.9. Both exact and heuristic algorithm are implemented by Python 3.7 and run on a ASUS Intel Core i7 with 1.8 GHz CPU and 8 GB RAM.


## Data
Both methodologies are first tested on the famous Solomon (1987) benchmark instances. The data is avalaible on [SINTEF](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/).

Then, the methods are applied to solve an logistics problem in a UK-based company.


## Methodology Summary

**1. Exact Method: Mathematical Formulations**

MIP Model: [View](https://github.com/dungtran209/Modelling-and-Analysis-of-a-Vehicle-Routing-Problem-with-Time-Windows-in-Freight-Delivery/blob/master/algorithm/VRPTW%20MIP%20Model.pdf)



**2. General Variable Neighbourhood Search (GVNS)**

Pseudo-code: [View](https://github.com/dungtran209/Modelling-and-Analysis-of-a-Vehicle-Routing-Problem-with-Time-Windows-in-Freight-Delivery/blob/master/algorithm/GVNS%20pseudo-code.png)

Flowchart: [View](https://github.com/dungtran209/Modelling-and-Analysis-of-a-Vehicle-Routing-Problem-with-Time-Windows-in-Freight-Delivery/blob/master/algorithm/GVNS%20flowchart.png)

Detailed elements of the GVNS:

Element | Content
------------ | -------------
Initial Solution Creation | Solomon I1 Heuristic and Clark & Wright Savings Heuristic
Improvement Operator | 2-opt, Or-opt, 2-opt*, Relocation, Exchange, CROSS, ICROSS, GENI, 𝜆-interchange
Local Search Process | Variable Neighbourhood Descent (Best-accept strategy)
Stopping Criteria | All neighbours (improvement operators) are explored


## Visualization

Best solution (minimum total distance of all tours) found by the GVNS for benchmark instance R206 (100 customers)
![Image](https://github.com/dungtran209/Modelling-and-Analysis-of-a-Vehicle-Routing-Problem-with-Time-Windows-in-Freight-Delivery/blob/master/data/1.%20Sample%20Dataset/R206.100.png)



