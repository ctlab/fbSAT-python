% constants
int: X; % number of input variables
int: D; % number of trace elements
int: P; % number of nodes
int: K; % number of transitions
int: k; % index of current transition

constraint assert(1 <= k /\ k <= K, "k should be in [1..K]");

% given data
array [1..D, 1..X] of bool: inputs; % input variables values before the i-th trace element is processed
array [1..D] of 0..K: tran_id;      % index of fired transition (0 - for non-fired)

% variables
array [1..P] of var 0..3: nodetype;
array [1..P] of var 0..X: terminal; % index of the input variable
array [1..P] of var 0..P: child;    % left child
array [1..P] of var 0..P: parent;   % parent index, or 0 if no parent
array [1..P, 1..D] of var bool: value;
array [1..P, 1..D] of var bool: child_value; % Left child value
array [1..P, 1..D] of var bool: child_value_second; % Right child value

var int: E; % number of edges
var int: V; % number of nodes

% constraints
% +++ 1.3. "AND"/"OR" typed nodes can't have number P or P-1
constraint nodetype[P] != 1 /\ nodetype[P] != 2;
constraint if P > 1
           then nodetype[P-1] != 1 /\ nodetype[P-1] != 2
           else true endif;
% +++ 1.4. "NOT" typed nodes can't have number P
constraint nodetype[P] != 3;

% 2.1. Only terminals have associated terminal variables
constraint forall (p in 1..P) (
    nodetype[p] = 0 <-> terminal[p] != 0
);

% 3.1. Root has no parent
constraint parent[1] = 0;
% 3.3. Non-root nodes must have a parent with lesser number
constraint forall (p in 2..P) (
    parent[p] != 0
    /\ parent[p] < p
);

% 4.1. Only terminals have no children
constraint forall (p in 1..P) (
    nodetype[p] = 0 <->
        child[p] = 0
);
% 4.2 "AND"/"OR" non-terminals have exactly 2 children, while left child has number from range [p+1 .. P-1]
constraint forall (p in 1..P-2) (
    nodetype[p] = 1 \/ nodetype[p] = 2 ->
        child[p] > p
        /\ child[p] < P
        /\ sum (c in p+1..P) (bool2int(parent[c] = p)) = 2
);
% 4.3. "NOT" non-terminals have exactly 1 child with greater number
constraint forall (p in 1..P-1) (
    nodetype[p] = 3 ->
        child[p] > p
        /\ sum (c in p+1..P) (bool2int(parent[c] = p)) = 1
);

% 5.1. Parent of left child of node p  *is*  p
constraint forall (p in 1..P, c in p+1..P) (
    child[p] = c -> parent[c] = p
);
% 5.2. Parent of right child of node p (AND/OR type)  *is*  p
constraint forall (p in 1..P, c in p+1..P-1) (
    (nodetype[p] = 1 \/ nodetype[p] = 2)
    /\ child[p] = c ->
        parent[c+1] = p
);
% 5.3. If vertex p is called a "parent", it is to have a child
constraint forall (p in 1..P, c in p+1..P) (
    parent[c] = p -> child[p] != 0
);

% 6.1. Terminal value
constraint forall (p in 1..P) (
    nodetype[p] = 0 ->
        forall (d in 1..D)
            (value[p, d] <-> inputs[d, terminal[p]])
);
% 6.2. AND value
constraint forall (p in 1..P) (
    nodetype[p] = 1 ->
        forall (d in 1..D)
            (value[p, d] <-> child_value[p, d] /\ child_value_second[p, d])
);
% 6.3. OR value
constraint forall (p in 1..P) (
    nodetype[p] = 2 ->
        forall (d in 1..D)
            (value[p, d] <-> child_value[p, d] \/ child_value_second[p, d])
);
% 6.4. NOT value
constraint forall (p in 1..P) (
    nodetype[p] = 3 ->
        forall (d in 1..D)
            (value[p, d] <-> not child_value[p, d])
);
% 6.5. not fired
constraint forall (d in 1..D) (
    tran_id[d] = 0 ->
        not value[1, d]
);
% 6.6. fired(k) -> not_fired(<k)
constraint forall (d in 1..D) (
    k < tran_id[d] ->
        not value[1, d]
);
constraint forall (d in 1..D) (
    k = tran_id[d] ->
        value[1, d]
);

% 7.1. Left child value
constraint forall (p in 1..P, c in p+1..P) (
    child[p] = c ->
        forall (d in 1..D)
            (child_value[p, d] <-> value[c, d])
);
% 7.2. Right child value
constraint forall (p in 1..P, c in p+1..P-1) (
    (nodetype[p] = 1 \/ nodetype[p] = 2)
    /\ child[p] = c ->
        forall (d in 1..D)
            (child_value_second[p, d] <-> value[c + 1, d])
);


% 8.1.
constraint E = sum (p in 1..P) (bool2int(parent[p] != 0));
% 8.2.
constraint V = P;
% 8.3.
constraint E = V - 1;


%array [1..P, 1..2] of var 1..P+1: child_orig;
%constraint forall (p in 1..P) (
%    nodetype[p] = 0 ->
%        child_orig[p, 1] = P + 1
%        /\ child_orig[p, 2] = P + 1
%);
%constraint forall (p in 1..P) (
%    nodetype[p] = 1 \/ nodetype[p] = 2 ->
%        child_orig[p, 1] = child[p]
%        /\ child_orig[p, 2] = child[p] + 1
%);
%constraint forall (p in 1..P) (
%    nodetype[p] = 3 ->
%        child_orig[p, 1] = child[p]
%        /\ child_orig[p, 2] = P + 1
%);


solve satisfy;


output ["K=", show(K), ", P=", show(P), ", D=", show(D), ", E=", show(E), "\n"] ++
       ["# nodetype = ", show(nodetype), "\n",
        "# terminal = ", show(terminal), "\n",
        "# parent = ", show(parent), "\n",
        %"# child = ", show(child_orig), "\n",
        "# left-child = ", show(child), "\n"]
