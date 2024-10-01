def compute_policy(alpha0, k, t, beta0, max_steps, gamma, c):
    V = {}
    action = {}
    alpha = alpha0

    # Initialize the value function at beta_max
    V[max_steps] = t
    action[max_steps] = 'take_over'

    # Compute the value function and optimal action for each beta from beta_max - 1 down to beta0
    for step in reversed(range(max_steps)):
        E_p = alpha / (alpha + step + beta0)
        V_next = V[step + 1]
        ContinueValue = -c + gamma * (E_p * k + (1 - E_p) * V_next)
        TakeOverValue = t

        if ContinueValue >= TakeOverValue:
            V[step] = ContinueValue
            action[step] = 'continue'
        else:
            V[step] = TakeOverValue
            action[step] = 'take_over'

    return V, action



def main():
    # Parameters
    c = 1      # Cost per time step
    gamma = 1  # Discount factor (less than 1)
    k = 20        # Utility if the child succeeds
    t = 0         # Utility if the parent takes over
    alpha0 = 2    # Initial alpha parameter of the Beta prior
    beta0 = 2     # Initial beta parameter of the Beta prior
    max_steps = 100  # Maximum beta value to prevent infinite loops
    
    V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
    
    print("Policy for up to max_steps")
    for step in range(max_steps):
        print(f"time: {step}, V: {V[step]:.4f}, action: {action[step]}")

if __name__ == "__main__":
    main()
