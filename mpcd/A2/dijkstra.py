F = lambda update: lambda state: (
    state if not state[0] 
    else (lambda z: (
        state[0] - {z}, 
        update(state[1], state[0] - {z}, z)
    ))(min(state[0], key=lambda v: state[1][v]))
)
make_update = lambda V, E, wt: (lambda d, U, v: {
    u: min(d[u], d[v] + wt((v, u))) 
    if (u in U) and ((v, u) in E) 
    else d[u] 
    for u in V
})
#loop = lambda f: (lambda g: g(g))(lambda g: lambda x: f(g(g)(x)))
def loop(F):
    def inner(state):
        while True:
            new_state = F(state)
            if new_state == state:  # Fixed point reached
                return state
            state = new_state
    return inner

pi = lambda X: X[1] # input is X = (U,d) where U is the set of unvisited vertices and d is the distance mapping

rho = lambda X: (set(X[0][0]), {v: 0 if v == X[1] else float('inf') for v in X[0][0]}) # input is X = (G, s) where G = (V, E, wt) is a directed graph and s is the start vertex

def mapcode(rho, F, pi):
    def f(i):
        x0 = rho(i)
        F_inf = loop(F) # F_inf = F^∞ , runs till fixed point
        xf = F_inf(x0)
        ans = pi(xf)
        return ans
    return f

def dijkstra(graph, s):
    V, E, wt = graph
    return mapcode(rho, F(make_update(V, E, wt)), pi)((graph, s))

# Example usage:
if __name__ == "__main__":
    # Define a directed graph1
    V = {0, 1, 2}
    E = {(0, 1), (1, 2), (0, 2)}
    wt = lambda e: {(0, 1): 1, (1, 2): 2, (0, 2): 4}[e]
    graph = (V, E, wt)
    
    distances = dijkstra(graph, 0)
    print(distances)  # Output: {0: 0, 1: 1, 2: 3}

    #6spbl
    #4cloverblack