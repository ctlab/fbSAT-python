% constants
int: P;  % number of nodes in parse tree
int: U;  % number of unique inputs
int: X;  % number of input variables

% data
array [1..U, 1..X] of bool: input_values;
array [1..U] of bool: root_value;

% variables
array [1..P] of var 0..3: nodetype;  % 0:terminal, 1:and, 2:or, 3:not
array [1..P] of var 0..X: terminal;  % index of the input variable
array [1..P] of var 0..P: child_left;
array [1..P] of var 0..P: child_right;
array [1..P] of var 0..P: parent;
array [1..P, 1..U] of var bool: value;
array [1..P, 1..U] of var bool: child_value_left;
array [1..P, 1..U] of var bool: child_value_right;

var int: E;  % number of edges
var int: V;  % number of nodes

% constraints
% 1. Parent constraints
% 1.1. Root has no parent
constraint parent[1] = 0;

% 1.2. BFS: typed nodes (except root) have parent with lesser number
constraint forall (p in 2..P) (
    parent[p] != 0 /\ parent[p] < p
);

% 1.3. parent<->child relation
constraint forall (p in 1..P-1, ch in p+1..P) (
    parent[ch] = p ->
        child_left[p] = ch \/ child_right[p] = ch
);

% 2. Terminals constraints
% 2.1. Only terminals have associated terminal variables
constraint forall (p in 1..P) (
    nodetype[p] = 0 <-> terminal[p] != 0
);

% 2.2. Terminals have no children
constraint forall (p in 1..P) (
    nodetype[p] = 0 ->
        child_left[p] = 0 /\ child_right[p] = 0
);

% 2.3. Terminals have value from associated input variable
constraint forall (p in 1..P, x in 1..X) (
    terminal[p] = x ->
        forall (u in 1..U) (value[p, u] <-> input_values[u, x])
);

% 3. AND/OR nodes constraints
% 3.0. AND/OR nodes cannot have numbers P-1 or P
constraint if P >= 1 then nodetype[P] != 1 /\ nodetype[P] != 2 else true endif;
constraint if P >= 2 then nodetype[P-1] != 1 /\ nodetype[P-1] != 2 else true endif;

% 3.1. AND/OR: left child has greater number
constraint forall (p in 1..P-2) (
    nodetype[p] = 1 \/ nodetype[p] = 2 ->
        child_left[p] > p /\ child_left[p] < P
);

% 3.2. AND/OR: right child is adjacent (+1) to left
constraint forall (p in 1..P-2) (
    nodetype[p] = 1 \/ nodetype[p] = 2 ->
        child_right[p] = child_left[p] + 1
);

% 3.3. AND/OR: children`s parents
constraint forall (p in 1..P-2, ch in p+1..P-1) (
    (nodetype[p] = 1 \/ nodetype[p] = 2) /\ child_left[p] = ch ->
        parent[ch] = p /\ parent[ch+1] = p
);

% 3.4a AND/OR: child_value_left is a value of left child
constraint forall (p in 1..P-2, ch in p+1..P-1) (
    (nodetype[p] = 1 \/ nodetype[p] = 2) /\ child_left[p] = ch ->
        forall (u in 1..U) (child_value_left[p, u] <-> value[ch, u])
);

% 3.4b AND/OR: child_value_right is a value of right child
constraint forall (p in 1..P-2, ch in p+2..P) (
    (nodetype[p] = 1 \/ nodetype[p] = 2) /\ child_right[p] = ch ->
        forall (u in 1..U) (child_value_right[p, u] <-> value[ch, u])
);

% 3.5a AND: value is calculated as a conjunction of children
constraint forall (p in 1..P-2) (
    nodetype[p] = 1 ->
        forall (u in 1..U) (value[p, u] <-> child_value_left[p, u] /\ child_value_right[p, u])
);

% 3.5b OR: value is calculated as a disjunction of children
constraint forall (p in 1..P-2) (
    nodetype[p] = 2 ->
        forall (u in 1..U) (value[p, u] <-> child_value_left[p, u] /\ child_value_right[p, u])
);

% 4. NOT nodes constraints
% 4.0. NOT nodes cannot have number P
constraint nodetype[P] != 3;

% 4.1. NOT: left child has greater number
constraint forall (p in 1..P-1) (
    nodetype[p] = 3 ->
        child_left[p] > p
);

% 4.2. NOT: no right child
constraint forall (p in 1..P-1) (
    nodetype[p] = 3 ->
        child_right[p] = 0
);

% 4.3. NOT: child`s parents
constraint forall (p in 1..P-1, ch in p+1..P) (
    nodetype[p] = 3 /\ child_left[p] = ch ->
        parent[ch] = p
);

% 4.4a NOT: child_value_left is a value of left child
constraint forall (p in 1..P-1, ch in p+1..P) (
    nodetype[p] = 3 /\ child_left[p] = ch ->
        forall (u in 1..U) (child_value_left[p, u] <-> value[ch, u])
);

% 4.4b NOT: child_value_right is False
constraint forall (p in 1..P-1, ch in p+1..P) (
    nodetype[p] = 3 ->
        forall (u in 1..U) (not child_value_left[p, u])
);

% 4.5. NOT: value is calculated as a negation of child
constraint forall (p in 1..P-1, ch in p+1..P) (
    nodetype[p] = 3 ->
        forall (u in 1..U) (value[p, u] <-> not child_value_left[p, u])
);

% 5. Root value
constraint forall (u in 1..U) (
    value[1, u] <-> root_value[u]
);

% 6. Tree constraints
% 6.1. Edges
constraint E = sum (p in 1..P) (bool2int(parent[p] != 0));
% 6.2. Vertices
constraint V = P;
% 6.3. Tree equality
constraint E = V - 1;


solve satisfy;


%output ["P=", show(P), ", U=", show(U), ", X=", show(X), "\n"] ++
%       ["# nodetype = ", show(nodetype), "\n",
%        "# terminal = ", show(terminal), "\n",
%        "# parent = ", show(parent), "\n",
%        "# left-child = ", show(child_left), "\n"]
