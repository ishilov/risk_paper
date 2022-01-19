set nodes;
set probas;

param D_min {i in nodes};
param D_max {i in nodes};

param G_min {i in nodes};
param G_max {i in nodes};

param kappa {i in nodes, j in nodes};

param cost {i in nodes, j in nodes};

param a_t {i in nodes};
param b_t {i in nodes};

param chi {i in nodes};
param p {j in probas};

param a {i in nodes};
param b {i in nodes};
param d {i in nodes};

param D_t {i in nodes, j in probas};
param G_d {i in nodes, j in probas};


var D {i in nodes, j in probas} >= D_min[i], <= D_max[i];
var G {i in nodes, j in probas} >= G_min[i], <= G_max[i];
var Q {i in nodes, j in probas};
var u {i in nodes, j in probas};
var eta {i in nodes};
var quant {i in nodes, j in nodes, k in probas}, >= -kappa[i,j], <= kappa[i,j];



minimize Total_cost:
	sum {i in nodes} (eta[i] + 1/(1-chi[i]) * sum {j in probas} p[j]*u[i,j]);
	
subject to Trade {i in nodes, j in nodes, k in probas}:
	quant[i,j,k] + quant[j,i,k] <= 0;
	
subject to Balance {i in nodes, j in probas}:
	D[i,j] == G[i,j] + G_d[i,j] + sum{k in nodes} quant[i,k,j];

subject to trading_sum {i in nodes, j in probas}:
	Q[i,j] = sum{k in nodes} quant[i,k,j];
	
subject to U {i in nodes, j in probas}:
	u[i,j] >= 0 ;
	
	
subject to Epigraph {i in nodes, j in probas}:
	a_t[i]*(D[i,j]-D_t[i,j])^2 - b_t[i] + 0.5*a[i]*G[i,j]*G[i,j] +  b[i]*G[i,j] + d[i]
	+ sum{k in nodes} cost[i,k] * quant [i,k,j] -eta[i] <= u[i,j];