# GlobalSwarming
Like global warming but with swarms

#example

``` matlab

%the function to minimize
obfunc = @(x) (x(1)^2-x(2))^(2) + (x(2)^2 + x(1) +8) + (x(1) - x(2) + 2*x(3))^2;

%the dimensions of the search space
dim = [-5 5;
       -5 5;
       -5 5];

SwarmSize = 10; %the number of particles to use
iterations = 100; %the number of iterations to execute
c1 = 2; %cognative scaling parameter
c2 = 2; % social scaling parameter
vmaxpercent = 0.2; %max velocity scaling parameter

%calling the particle swarm optimizer
[bestPos,bestVal] = PSO(obfunc,dim,SwarmSize,c1,c2,iterations,vmaxpercent)
```
