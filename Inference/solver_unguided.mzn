include "globals.mzn";
include "alldifferent.mzn";

% inputs
int: start;
int: end;
int: number_of_nodes;
array[1..number_of_nodes+2, 1..number_of_nodes+2] of int: matrix;

% output
array[1..number_of_nodes+2] of var 1..number_of_nodes+2: succ;

% constraints
constraint alldifferent(succ);
constraint succ[end] == start;
constraint circuit(succ);

% cost function
var int: sum_path;
constraint sum_path = sum(i in 1..number_of_nodes+2 where i != end) (matrix[i,succ[i]]);
  
% solving
solve minimize sum_path;

