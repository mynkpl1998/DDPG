import gymnasium as gym

if __name__ == "__main__":
    
    # Test Agent
    num_test_episodes = 1
    test_env = env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")

    for episode in range(0, num_test_episodes):

        test_env.reset()
        # Keep track of num of steps in episode
        num_steps = 0
        
        done = False
        while not done:
            test_env.render()
            action = test_env.action_space.sample()
            observation, reward, terminated, truncated, info = test_env.step(action)

            num_steps += 1
            
            if terminated or truncated:
                done = True
            time.sleep(0.5)

        print("Episodes: {}, Num Steps: {}".format(episode+1, num_steps))
    test_env.close()