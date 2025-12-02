#import "../lib/style.typ": *
#import "../lib/mapcode.typ": *

== Optimal Matrix Chain Multiplication
#set math.equation(numbering: none)

Given a sequence of matrices $A_(n_0 times n_1),A_(n_1 times n_2),..,A_(n_(q-1) times q_i)$, Compute the minimum number of scalar multiplication required to multiply all of them.  
The algorithm takes in vector $d = (n_0,..,n_q)$ and returns an integer.

Formal definition: $"num_mul"(d) := T_d (0,|d|-2)$ where

$
  T_d (i, j) = min_{k|i<=k<j} T_d (i,k)+T_d (k+1,j)+d(i) dot d(j+1) dot d(k+1)
$  
This is a classic dynamic programming algorithm where we build up by solving subproblems. $T_d$ is a matrix whose (i,j)th represents the subproblem instance (n_i,...,n_j+1)

Examples:
- $"num_mul"(10, 20, 30, 40, 50) -> 38000 $
- $"num_mul"(13, 5, 89, 3, 34) -> 2856$

*As mapcode:*

_primitives_: $"min",+,<,<=$. \
Non Comparitive Operations on $bot$ return $bot$. (For example: $4+bot=bot$ , $min({3,2,bot,10})=bot$)

$
  k in NN \
  I = NN^k quad quad quad X & = NN times NN -> NN_bot quad quad quad
  A = NN \
    rho(d) & = {(i,j) -> bot | i,j in NN} \
        F_d (x)(i,j) & = cases(
                  0 & "if " i = j,
                  x(i,j) & "if " x(i,j) != bot,
                  min { x(i,k)+ x(k+1,j)+d(i) dot d(j+1) dot d(k+1) |i<=k<j} & "else"
                ) \
        pi_d (x) & = x(0,|d|-2)
$

*Bound Function for Mapcode:*

$sigma : X -> NN$ such that 
$
  sigma(x)= |d-1|^2 - |{(i,j) in NN times NN : x(i,j) != bot}|
$
Statement : $forall x in X, x != F x => sigma(F x) < sigma(x)$

Proof: For any $x in X$ such that $x != F x$ ,

Without loss of generality, assume that $x != rho(i)$ for an $i in I$.

Case 1: If $exists (i,i) in NN times NN$ such that $ x(i,j)=bot$

Then $F x (i,i) = 0$ $because$ definition of $F$

Number of $bot$'s decrease. ie. $sigma(F x) < sigma(F x)$

Case 2: Else

Say $exists (i,j)$ within the grid with $x(i,j) = bot$ such that $forall k in NN, i <= k < j => x(i,k) != bot and x(k+1,j) != bot$

proof of existence. By Contradiction: Say there is no such (i,j) within the grid.

Then for any $(i,j)$ such that $x(i,j)$ within the grid, there exists a $i<=k<j$ such that either $x(i,k)$ or $x(k+1,j)$ is $bot$.
So from the definition of $F$, we will have $F x (i,j) = bot$ which is equal to $x(i,j)$

Therefore, $forall (i,j)$ within the grid, we obtain that $x(i,j)=F x (i,j)$. This means $x$ is a fixed point which contradicts our previous assertion that $x != F x$.  

 DONE

Now, since we have shown that such $(i,j)$ exists, from the definition of $F$, we get that $F(i,j) != bot$. 

So the number of $bot$'s decrease. ie. $sigma(F x) < sigma(F x)$

QED



*Invariant Functions for Mapcode:*

Let$
F^infinity (x) & = cases(
                  F x & "if " x = F x,
                  F^infinity (F x)& "else"
                ) \
$
Since $sigma$ is well founded, $F^infinity$ exists and maps $X -> "Fix"(F)$.

So we may define an invariant $Phi : X -> A$ such that 
$
  Phi(x)= pi compose F^infinity (x)
$

$forall x in X, Phi (F x) = Phi (x)$

proof: $Phi (F x) = pi compose F^infinity (F x) = pi (F (F^infinity x)) = pi (F^infinity x) = pi compose F^infinity (x) = Phi(x)$


$forall x in X, F x = x => Phi (x) = pi (x)$

proof: $Phi (x) = pi compose F^infinity (x) = pi(F^infinity (x)) = pi (x)$

$forall i in I, Phi (rho (i)) = pi compose F^infinity compose rho (i)$

proof: $Phi (rho (i)) = pi compose F^infinity (rho i) = pi compose F^infinity compose rho (i)$


#let inst_dimvecs  = (13, 5, 89, 3, 34);
#let inst_dm = inst_dimvecs.len();


#figure(
  caption: [Optimal Matrix Chain Multiplication for dimensions = #inst_dimvecs.],
$#{
  let rho = (inst_dimvecs) => {
    let x = ()
    for i in range(0, inst_dimvecs.len() - 1) {
      let row = ()
      for w in range(0, inst_dimvecs.len() - 1) {
        row.push(none)
      }
      x.push(row)
    }
    x
  }

  let safe_operation(a, b, c, op) = {
    if a == none or b == none or c == none {
      none
    } else {
      op(a, b, c)
    }
  }

  let F_i(d) = (x) => ((i, j)) => {
    if i == j {
      0
    } else {
      let min_val = none
      let mut = none
      for k in range(i, j) {
        let current = safe_operation(
          x.at(i).at(k),
          x.at(k+1).at(j), 
          d.at(i) * d.at(j+1) * d.at(k+1),
          (a, b, c) => a + b + c
        )
        
        if current != none {
          if min_val == none or current < min_val {
            min_val = current
          }
        } else {
          // If we want to propagate none immediately:
          // return none
          // If we want to check all k but return none if any is none:
          min_val = none
        }
      }
      min_val
    }
  }
  

  let F = (inst_dimvecs) => map_tensor(F_i(inst_dimvecs), dim: 2)

  let pi = ((dimvec)) => (x) => {
    let m = dimvec.len()
    x.at(0).at(m - 2)
  }

  // (2, 3, 4, 5), (3, 4, 5, 6), 5)
  let I_h(inst_dimvecs) = {
    [
      $d: vec(..#inst_dimvecs.map(i => [#i]), delim: "[")$
    ]
  }

  
  let x_h(x, diff_mask:none) = {
    set text(weight: "bold")
    let rows = ()

    
    let header_cells = ()
    header_cells.push(rect(fill: green.transparentize(70%), inset: 14pt)[$id$])
    

    for w in range(0, x.at(0).len()) {
      header_cells.push(rect(fill: orange.transparentize(70%), inset: 14pt)[$#w$])
    }
    rows.push(grid(columns: header_cells.len() * (30pt,), rows: 14pt, align: center + horizon, ..header_cells))

    for i in range(0, x.len()) {
      let row = ()
      
      if i == 0 {
        row.push(rect(fill: green.transparentize(70%), inset: 14pt)[$0$])
      }else{
        row.push(rect(fill: green.transparentize(70%), inset: 14pt)[$#i$])
      }

      for w in range(0, x.at(i).len()) {
        let val = if x.at(i).at(w) != none {[$#x.at(i).at(w)$]} else {[$bot$]}
        if diff_mask != none and diff_mask.at(i).at(w) {
          row.push(rect(stroke: gray, fill: yellow.transparentize(70%), inset: 14pt)[$#val$])
        } else {
          row.push(rect(stroke: gray, inset: 14pt)[$#val$])
        }
      }

      rows.push(grid(columns: row.len() * (30pt,), rows: 14pt, align: center + horizon, ..row))
    }

    grid(align: center, ..rows)
  }

  mapcode-viz(
    rho, F(inst_dimvecs), pi(inst_dimvecs),
    I_h: I_h,
    X_h: x_h,
    F_name: [$F_(w,v)$],
    pi_name: [$\pi_(w,v,C)$],
    group-size: calc.min(3, inst_dimvecs.len()),
    cell-size: 60mm, scale-fig: 75%
  )(inst_dimvecs)
}$)

