# Comedian Timetable Scheduler
This project schedules comedians for a comedy club, matching them with demographics based on preferences while minimising costs. 

## How It Works
### Basic Scheduling
This task is a constraint **satisfaction** problem where comedians are assigned to demographics for weekly shows. The solution employs:

* Backtracking Search: Assigns demographics to comedians based on theme matches.
* Heuristics:
  * Minimum Remaining Values (MRV): Chooses demographics with the fewest valid comedians to narrow the search space.
  * Least Constraining Value (LCV): Orders comedian choices to maximize future flexibility.
  * Forward Checking: Prunes invalid future assignments to enhance efficiency.

### Cost-Effective Scheduling
In this constraint **optimisation** problem, the goal is to minimize the cost of hiring comedians. The approach starts with a greedy schedule and 
iteratively explores changes to reduce costs, accepting worse solutions initially to escape local minima. This process is known as simulated annealing.
