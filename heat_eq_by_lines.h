#pragma once
#include<cmath>
#include<iostream>
#include<vector>

using namespace std;

/*
k is the constant in the heat equation u_t = k*u_xx.

This only simulates for dirichlet boundary conditions, the vector u0 should contain only the
initial values of the INTERIOR. Use the remaining parameters to specify your boundary values.
the program will automatically identify the value of the spatial grid spacing, dx.
*/

class heat_lines_sim
{
public:
    heat_lines_sim(double k, vector<double> u0, double a_, double b, double ua, double ub);

    void step(double dt);                   // run a single step of the simulation and update u

    void run_till(double tf, double dt);     // run until simulation reacher time t and update u

    void run_for(int n, double dt);         // run simulation for set number of steps and update u

    vector<double> get_state();

    double get_time();

private:
    vector<double> u;
    double k, t, a, b, ua, ub, dx, dx2;             // simulation parameters
    int n;


};