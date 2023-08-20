import cloudpickle

n_step_file = "n_step.pkl"
orig_file = "orig.pkl"

# Open the cloudpickle file
with open(n_step_file, 'rb') as f:
  # Load the object from the file
  n_step_data = cloudpickle.load(f)


# Open the cloudpickle file
with open(orig_file, 'rb') as f:
  # Load the object from the file
  orig_data = cloudpickle.load(f)

assert len(n_step_data) == len(orig_data)

for idx in range(len(n_step_data)):
  print(n_step_data[idx].n_step_reward, orig_data[idx].n_step_reward)