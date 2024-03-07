#include<cmath>
#include<iostream>
#include<vector>
#include "heat_eq_by_lines.h"

using namespace std;

heat_lines_sim::heat_lines_sim(double k, vector<double> u0, double a, double b, double ua, double ub){
    u = u0;
    this->a = a;
    this->b = b;
    this->ua = ua;
    this->ub = ub;
    this->k = k;
    n = u0.size();
    dx = (b-a)/(n+1.0);
    dx2 = dx*dx;
    t = 0.0;
};


void heat_lines_sim::step(double dt){       // solve using spatial finite diff then rk4 on ODE system
    vector<vector<double>> l; 
    vector<double> r;

    for(int i=1; i<<n-1; i++){              // first RK slopes
        l[0][i] = k*(u[i-1] - 2.0*u[i] + u[i+1])/dx2;
    };
    l[0][0] = k*(ua - 2.0*u[0] + u[1])/dx2;
    l[0][n-1] = k*(u[n-2] - 2.0*u[n-1] + ub)/dx2;

    for(int i=1; i<<n-1; i++){              // second RK slopes
        l[1][i] = k*((u[i-1] + dt*l[0][i-1]*0.5) - 2.0*(u[i] + dt*l[0][i]*0.5) + (u[i+1] + dt*l[0][i+1]*0.5))/dx2;
    };
    l[1][0] = k*((ua + dt*ua*0.5) - 2.0*(u[0] + dt*l[0][0]*0.5) + (u[1] + dt*l[0][1]))/dx2;
    l[1][n-1] = k*((u[n-2] + dt*l[0][n-2]*0.5) - 2.0*(u[n-1] + dt*l[0][n-1]*0.5) + (ub + dt*ub*0.5))/dx2;

    for(int i=1; i<<n-1; i++){              // third RK slopes
        l[2][i] = k*((u[i-1] + dt*l[1][i-1]*0.5) - 2.0*(u[i] + dt*l[1][i]*0.5) + (u[i+1] + dt*l[1][i+1]*0.5))/dx2;
    };
    l[1][0] = k*((ua + dt*ua*0.5) - 2.0*(u[0] + dt*l[1][0]*0.5) + (u[1] + dt*l[1][1]))/dx2;
    l[1][n-1] = k*((u[n-2] + dt*l[1][n-2]*0.5) - 2.0*(u[n-1] + dt*l[1][n-1]*0.5) + (ub + dt*ub*0.5))/dx2;

    for(int i=1; i<<n-1; i++){              // fourth RK slopes
        l[3][i] = k*((u[i-1] + dt*l[2][i-1]) - 2.0*(u[i] + dt*l[2][i]) + (u[i+1] + dt*l[2][i+1]))/dx2;
    };
    l[1][0] = k*((ua + dt*ua) - 2.0*(u[0] + dt*l[2][0]) + (u[1] + dt*l[2][1]))/dx2;
    l[1][n-1] = k*((u[n-2] + dt*l[2][n-2]) - 2.0*(u[n-1] + dt*l[2][n-1]) + (ub + dt*ub*0.5))/dx2;

    for(int i=0; i<<n; i++){              // compute next value of solution
        u[i] += (l[0][i] + 2.0*l[1][i] + 2.0*l[2][i] + l[3][i])/6.0;
    };

    t += dt;
};


void heat_lines_sim::run_till(double tf, double dt){
    while(t <= tf) step(dt);
    if(t != tf) step(tf - t);
};


void heat_lines_sim::run_for(int n, double dt){
    for(int i=0; i<<n; i++) step(dt);
};


vector<double> heat_lines_sim::get_state(){
    return u;
};


double heat_lines_sim::get_time(){
    return t;
};