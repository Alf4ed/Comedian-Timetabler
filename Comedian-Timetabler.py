import comedian
import demographic
import ReaderWriter
import timetable
import random
import math

class Scheduler:

	def __init__(self, comedian_List, demographic_List):
		self.comedian_List = comedian_List
		self.demographic_List = demographic_List
	#The five calls you can use are timetableObj.addSession, d.reference, d.topics, c.name, c.themes

	"""	
		All three tasks are constraint satisfaction problems. Task 1 introduces the basic concept,
		Task 2 introduces more constraints and increases the search space. Task 3 introduces the concept
		of costs, and therefore Task 3 is a constraint optimisation problem.

		My solution to Task 1 uses a backtracking search, along with heuristics and problem specific knowledge
		to solve the problem of scheduling main shows.

		My first approach was to make each day and slot a variable, and then assign demographic-comedian pairs
		to each of these variables. However, I realised that this would allow for backtracking to different slots
		on the same day. This is wasted computation as the order of shows on each day does not effect any of the 
		constraints. I therefore changed my approach to assign comedian-day values to demographic variables. This
		reduces the seach space significantly.

		When choosing the order to select variables (demographics), I use the Minimum Remaining Values heuristic. I
		therefore chose variables with the fewest different potential values. In practice, this causes the program to 
		"fail early" - the program picks variables that are most likely to cause a failure. To implement this heuristic,
		I count the total number of comedian-day values that are consistent with each variable, then choose the variable
		that has the fewest of these values.

		I considered also using the "Degree Heuristic", however every slot has constraints linking it to all other slots.
		Therefore the degree heuristic cannot be applied in this situation.

		After choosing a variable, next I used the Least Constraining Value heuristic to order the values that can be
		assigned to that variable. In practice, this means that values that cannot be marketed to future demographics 
		are assigned first. This is because these comedians reduce the number of values that can be assigned to future 
		demographics by the smallest amount. This heuristic aims to maximise the choice available for future variable-value
		assignments.

		When selecting values, Forward checking is used to eliminate any values from the domain of unassigned variables that
		are known to be inconsistent with the current assignment. This helps to avoid backtracking and improve the efficiency
		of the search. Because of the pruning that takes place, chronological backtracking is effective and the need for the 
		more complex approach of backjumping is reduced.
	"""	

	def isConsistent(self, assignment, comedian, day):
		# Returns False if the comedian has already performed 2 times in the week
		# Returns False if the comedian has already performed that day
		showsPerWeek = 0
		maxOnDay = 0
		for show in assignment:
			if show[0] == day:
				maxOnDay += 1
				if maxOnDay >= 5:
					return False
			if show[1] == comedian:
				showsPerWeek += 1
				if showsPerWeek >= 2 or show[0] == day:
					return False

		return True
	
	# Uses MRV to decide the order of selection of variables
	def selectUnassignedVariable(self, assignment, slots):
		days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
		variables = []

		for showLeft in slots:
			count = 0
			for comedian in self.comedian_List:
				if all(item in comedian.themes for item in showLeft.topics):
					for d in days:
						if self.isConsistent(assignment, comedian, d):
							count += 1
			variables.append([showLeft, count])
		
		# Chooses the variable that has the fewest possible values
		tempVariable = []
		min = 0
		for i in variables:
			if tempVariable == [] or i[1] < min:
				tempVariable = i[0]
				min = i[1]

		slots.remove(tempVariable)

		return tempVariable

	# Uses the Least Constraining Value search Heuristic to order the values that are tested
	def orderDomainValues(self, assignment, variable, slots):
		values = []
		days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

		for d in days:
			for c in self.comedian_List:
				if all(item in c.themes for item in variable.topics):
					if self.isConsistent(assignment, c, d):

						count = 0
						for demographic in slots:
							if all(item in c.themes for item in demographic.topics):
								count += 1

						values.append([c,d,count])
		
		# Order the values, starting with the comedians that can be marketed to the fewest subsequent slots
		values.sort(key=lambda row: row[2])

		return values

	def backtrack(self, assignment, slots):
		# Returns the schedule if all slots have been assigned values
		if len(slots) == 0:
			return assignment
		
		# Uses the MRV heuristic to choose the order to pick variables
		variable = self.selectUnassignedVariable(assignment, slots)
		# Uses the LCV heuristic to chose the order to assign values
		values = self.orderDomainValues(assignment, variable, slots)

		for val in values:
			if self.isConsistent(assignment, val[0], val[1]):
				assignment.append([val[1], val[0], variable])
				result = self.backtrack(assignment, slots)
				if result is not None:
					return result
				assignment.pop()

		slots.insert(0, variable)
		return None

	#This method should return a timetable object with a schedule that is legal according to all constraints of Task 1.
	def createSchedule(self):
		#Do not change this line
		timetableObj = timetable.Timetable(1)

		#Here is where you schedule your timetable
		days = {"Monday": 1, "Tuesday": 1, "Wednesday": 1, "Thursday": 1, "Friday": 1}
		assignment = []
		slots = []		
		
		for demographic in self.demographic_List:
			slots.append(demographic)
		
		result = self.backtrack(assignment, slots)

		for show in result:
			timetableObj.addSession(show[0], days[show[0]], show[1], show[2], "main")
			days[show[0]] += 1

		#Do not change this line
		return timetableObj

	"""	
		Task 2 introduces more constraints, and also increases the problem size. The general approach to solving this 
		problem however is very similar to that of Task 1.

		As a result of the increased problem size, the heuristics that are used need to be effective in order to solve the 
		problems in a reasonable amount of time. Like for task 1, a backtracking search is used to find a valid assignment.

		A small change from Task 1 is that the variables used in the backtracking are no longer demographics. Now, each
		demographic needs both a main and test show - the variables are therefore now demographic-showType pairs.

		The function "isConsistentTask2()" is used to check that given an existing assignment, a value can be assined to a
		new variable. This function is used to check the consistency of the timetable throughout the backtracking search.

		The main difference between Task 2 and Task 1 is the introduction of test shows. Previously, I have used the "all()"
		function to check that a comedian's main show can be marketed to a demographic. For Task 2 and 3, I now use either the
		"all()" function, or the "any()" function depending on whether the show is a main or test show. This enforces the two
		different constraints outlined in the specification. Additonally, 10 shows are now allowed on each day, rather than 5.

		The MRV heuristic is still used to order the different variables. The number of potential values that can be assigned to
		the variable is calculated differently depending on whether a main or test show is being considered. The constraints on test
		shows are more relaxed and therefore test shows tend to have more remaining values.
	"""

	def isConsistentTask2(self, assignment, showType, comedian, day):
		# Returns False if the comedian would perform for more than 4 hours in the week
		# Returns False if the comedian would perform for more than 2 hours in a day
		# Returns False if any day has more than 10 slots assigned
		showCosts = {"main": 2, "test": 1}
		showsPerWeek = 0
		showsPerDay = 0
		maxOnDay = 0

		showsPerWeek += showCosts[showType]
		showsPerDay += showCosts[showType]

		for show in assignment:
			if show[0] == day:
				maxOnDay += 1
				if maxOnDay >= 10:
					return False
			if show[1] == comedian:
				showsPerWeek += showCosts[show[3]]
				if showsPerWeek > 4:
					return False

				if show[0] == day:
					showsPerDay += showCosts[show[3]]
					if showsPerDay > 2:
						return False

		return True
	
	# Uses the Most Constraining Value heuristic to chose the next variable to assign values to
	def selectUnassignedVariableTask2(self, assignment, slots):
		days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
		variables = []

		# For each variable that is still unnassigned, calculate the total number of values that could be assigned
		for showLeft in slots:
			count = 0
			if showLeft[1] == "main":
				for comedian in self.comedian_List:
					if all(item in comedian.themes for item in showLeft[0].topics):
						for day in days:
							if self.isConsistentTask2(assignment, showLeft[1], comedian, day):
								count += 1
			if showLeft[1] == "test":
				for comedian in self.comedian_List:
					if any(item in comedian.themes for item in showLeft[0].topics):
						for day in days:
							if self.isConsistentTask2(assignment, showLeft[1], comedian, day):
								count += 1
			variables.append([showLeft[0], showLeft[1], count])

		# Find the variable that has the smallest choice of what values can be assigned
		tempVariable = []
		min = 0
		for i in variables:
			if tempVariable == [] or i[2] < min:
				tempVariable = [i[0], i[1]]
				min = i[2]

		slots.remove([tempVariable[0], tempVariable[1]])

		return tempVariable

	# Uses a greedy rule to order the values
	def orderDomainValuesTask2(self, assignment, variable):
		days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
		values = []

		# Create a list that stores all of the values that can be assigned to the specified variable
		if variable[1] == "main":
			for c in self.comedian_List:
				if all(item in c.themes for item in variable[0].topics):
					for d in days:
						if self.isConsistentTask2(assignment, "main", c, d):
							values.append([c,d])

		if variable[1] == "test":
			for c in self.comedian_List:
				if any(item in c.themes for item in variable[0].topics):
					for d in days:
						if self.isConsistentTask2(assignment, "test", c, d):
							values.append([c,d])

		# Calculate the cost of the schedule formed by assigning the specified variable each of the possible values
		orderedValues = []
		for c, d in values:
			assignment.append([d, c, variable[0], variable[1]])
			orderedValues.append([c, d, self.costOfAssignment(assignment)])
			assignment.pop()

		# Sort the list in increasing order of cost
		# If two values result in the same schedule cost, chose randomly
		orderedValues.sort(key=lambda row: (row[2], random.random()))

		values = []
		for i in orderedValues:
			values.append([i[0], i[1]])

		return values

	def backtrackTask2(self, assignment, slots):
		# Returns the schedule if all slots have been assigned values
		if len(slots) == 0:
			return assignment
		
		# Uses MRV to chose a variable
		variable = self.selectUnassignedVariableTask2(assignment, slots)
		# Uses a greedy rule to order the values that can be assigned to the variable
		values = self.orderDomainValuesTask2(assignment, variable)

		for val in values:
			if self.isConsistentTask2(assignment, variable[1], val[0], val[1]):
				assignment.append([val[1], val[0], variable[0], variable[1]])
				result = self.backtrackTask2(assignment, slots)
				# If a solution has been found, then return the schedule
				if result is not None:
					return result
				assignment.pop()

		slots.insert(0, variable)
		return None

	#This method should return a timetable object with a schedule that is legal according to all constraints of Task 2.
	def createTestShowSchedule(self):
		#Do not change this line
		timetableObj = timetable.Timetable(2)

		#Here is where you schedule your timetable
		days = {"Monday": 1, "Tuesday": 1, "Wednesday": 1, "Thursday": 1, "Friday": 1}
		assignment = []
		slots = []
		
		for demographic in self.demographic_List:
			slots.append([demographic, "main"])
			slots.append([demographic, "test"])
		
		result = self.backtrackTask2(assignment, slots)

		for show in result:
			timetableObj.addSession(show[0], days[show[0]], show[1], show[2], show[3])
			days[show[0]] += 1

		#Do not change this line
		return timetableObj

	"""
		Task 3 introduces costs to the problem. The aim of this solution is therefore to find a legal solution to
		each problem, whilst also minimising an objective cost function.

		In order to solve this optimisation problem, I considered a number of different approaches. The general form
		of these approaches can be divided into two main catagories: informed search and local search.

		Informed search:
			I considered both A* algorithm and Depth First Branch and Bound.
			A* algorithm uses a heuristic function that estimates future cost. The algorithm then assigns values to
			variables in increasing order of the current cost + the estimated future cost.

			Depth First Branch and Bound uses a similar approach to the backtracking algorithm. However, instead of 
			terminating the search after a solution is found, the algorithm continues the search. At each stage, if
			the current cost + the estimated future cost of a solution is greater than the cost of the best solution
			already found, this branch can be pruned, reducing the search space.

			Both of these algorithms need an admissible heuristic that is an underestimate of the future cost of completing
			an assinment. Finding such a heuristic proved a problem, as if the estimate is too low, very few branches are 
			pruned and the problem takes a great amount of time to find a solution.
		
		After ruling out informed search, I decided to use a local search technique. A number of different approaches
		were considered: (Constraint-based Timetabling, Tomáš Müller)
			Hill climbing, randomized greedy descent, random walk and simulated annealing were all considered.

			Of these different approaches, I concluded that a stochastic approach would be best, as this helps to avoid
			local minima and plateauxs. Simulated annealing allows for the amount of randomness to decrease over time, 
			which allows the program to converge on the optimal solution. This is therefore the approach that was chosen.
		
		If the cost of a solution is reduced after making a change to the values, this new solution is adopted as it is closer
		(in terms of cost) to the optimal solution. An acceptance function is used to outline the probability that a new solution that is worse than the previous one
		is accepted. I used the probability function described by Kirkpatrick et al., defined as exp(-(c'-c)/T). A geometric
		cooling schedule is then used to reduce the temperature value after each iteration (A Comparison of Cooling Schedules
		for Simulated Annealing (Artificial Intelligence), what-when-how.com). The initial temperature value was
		chosen to be scaled to the values in the problem, and the decrease in temperature was adjusted considering the number
		of iterations during each run. More worsoning changes are therefore made at the begining of the programs run.

		In order to improve the efficiency of the simulated annealing process, a greedy algorithm is used to calculate an 
		innitial schedule. This algorithm works in place of the LCV heuristic, ordering values for a variable in increasing order
		or cost of the schedule resulting from assigning these values to the variable specified. Few changes are therefore needed 
		to reduce the schedule to a near optimal solution. Furthermore, the annealing process is restarted 4 times, and only the 
		best schedule is returned. This was used instead of a parallel search or a stochastic beam search.

		A 1 element taboo list was also used, which means that any changes to the schedule during the annealing process
		produced a different schedule - in practice this meant that the elements on two different days were switched (as 
		switching shows on the same day would have no effect on the constraints/cost)

		Finally, specific problem knowledge was used to terminate the program early if a solution was found with optimal cost.
		Optimal cost for this problem is 10050. No schedule is able to have a cost less than this amount, and any changes after 
		finding such a schedule are a waste of computational time.
	"""

	def costOfAssignment(self, assignment):
		totalCost = 0

		dayBefore = {"Monday": None,
					"Tuesday": "Monday",
					"Wednesday": "Tuesday",
					"Thursday": "Wednesday",
					"Friday": "Thursday"}

		slots = {"Monday": {"main": [], "test": []},
				"Tuesday": {"main": [], "test": []},
				"Wednesday": {"main": [],"test": []},
				"Thursday": {"main": [], "test": []},
				"Friday": {"main": [], "test": []}}

		for show in assignment:
			day = slots[show[0]]
			showType = day[show[3]]
			showType.append(show[1])
		
		comedianMainShows = dict()
		comedianTestShows = dict()

		for day, values in slots.items():
			for showType, comedians in values.items():
				if showType == "main":
					for comedian in comedians:
						if comedian not in comedianMainShows:
							comedianMainShows[comedian] = 1
							totalCost += 500
						else:
							comedianMainShows[comedian] += 1
							# All main shows on Monday must be at full price as there is no day before Monday
							if day != "Monday" and comedian in slots[dayBefore[day]][showType]:
								totalCost += 100
							else:
								totalCost += 300
				if showType == "test":
					for comedian in comedians:
						if comedian not in comedianTestShows:
							comedianTestShows[comedian] = 1
						else:
							comedianTestShows[comedian] += 1

						# If the comedian is performing twice on the same day, the cost of test shows is halved
						if slots[day][showType].count(comedian) >= 2:
							totalCost += (300 - (50 * comedianTestShows[comedian])) / 2
						else:
							totalCost += 300 - (50 * comedianTestShows[comedian])

		return totalCost

	def legalScheduleTask3(self, assignment, a, b):
		showCosts = {"main": 2, "test": 1}

		for i in [a, b]:
			showsPerDay = 0
			showsPerWeek = 0
			for show in assignment:
				# The comedian matches
				if show[1] == assignment[i][1]:
					# The day matches
					if show[0] == assignment[i][0]:
						showsPerDay += showCosts[show[3]]
					
					showsPerWeek += showCosts[show[3]]
		
					if showsPerDay > 2 or showsPerWeek > 4:
						return False

		return True
	
	def simulatedAnnealing(self, assignment, bestSchedule):
		# We run the annealing process for 10,000 iterations
		changesLeft = 10000
		temperature = 100
		coolingRate = 0.01

		prevCost = self.costOfAssignment(assignment)
		bestScheduleCost = self.costOfAssignment(bestSchedule)

		for iterations in range(0, changesLeft):
			a = random.randint(0, 49)
			b = random.randint(0, 49)

			originalComedian = assignment[a][1]

			# Ensure that the shows we are switching are on different days
			# Switching shows on the same day will have no effect to the total cost
			while assignment[a][0] == assignment[b][0]:
				b = random.randint(0, 49)
			
			# At each stage in the annealing process, swap one comedian with a different comedian who can be marketed to the demographic
			# There might not always be such a comedian, so it is possible for a comedian to be switched with itself (no change)
			legal = False
			while not legal:
				assignment[a][1] = random.choice(self.comedian_List)
#
				if assignment[a][3] == "main":
					legal = all(item in assignment[a][1].themes for item in assignment[a][2].topics)
				if assignment[a][3] == "test":
					legal = any(item in assignment[a][1].themes for item in assignment[a][2].topics)

			# At each stage in the annealing process, swap the days of two demographic-comedian-showtype triples
			tempDay = assignment[a][0]
			assignment[a][0] = assignment[b][0]
			assignment[b][0] = tempDay

			# This is the new cost of the schedule after making the adjustments
			newCost = self.costOfAssignment(assignment)

			# If the schedule is legal and the changes made have reduced the cost of the schedule, then adopt it
			# If the schedule's cost has increased, adopt the new schedule probabilistically based on the increased cost and a temperature function
			if self.legalScheduleTask3(assignment, a, b) and (newCost <= prevCost or (math.exp((prevCost - newCost) / temperature) > random.uniform(0, 1))):
				prevCost = newCost
			# If we do not adopt the new schedule, then revert any changes that have been made
			else:
				tempDay = assignment[a][0]
				assignment[a][0] = assignment[b][0]
				assignment[b][0] = tempDay

				assignment[a][1] = originalComedian

			# A geometric cooling function is used to reduce the temperature value after each iteration
			temperature *= (1-coolingRate)

			# Store the current best known schedule
			# If the annealing process makes the solution worse, we still return the best solution known
			if prevCost < bestScheduleCost:
				bestSchedule = assignment.copy()
				bestScheduleCost = prevCost

			# If a solution with cost 10050 (an optimal solution) is found, the annealing process can terminate early
			if prevCost <= 10050:
				break

		return bestSchedule, bestScheduleCost

	def createMinCostSchedule(self):
		#Do not change this line
		timetableObj = timetable.Timetable(3)
				
		#Here is where you schedule your timetable
		days = {"Monday": 1, "Tuesday": 1, "Wednesday": 1, "Thursday": 1, "Friday": 1}
		assignment = []

		# The slots array is used to store the variables that are yet to be assigned values
		slots = []
		for demographic in self.demographic_List:
			slots.append([demographic, "main"])
			slots.append([demographic, "test"])

		# 20000 is greater than the maximum possible cost of a solution
		optimalCost = 20000
		optimalAssignment = None

		# Before running the annealing process, use the task 2 solver to find a valid timetable
		greedyAssignment = self.backtrackTask2(assignment.copy(), slots.copy())

		# The annealing process is run 4 times, and the lowest cost timetable is chosen
		for i in range(0, 4):
			annealedAssignment, cost = self.simulatedAnnealing(greedyAssignment.copy(), greedyAssignment.copy())
			
			# If the solution to this restart of the annealing process is better than the previous optimal solution, then update the optimal value
			if cost < optimalCost:
				optimalAssignment = annealedAssignment
				optimalCost = cost
			
			# If a solution is found with the optimal cost, there is no need to run multiple restarts
			if cost <= 10050:
				break

		for show in optimalAssignment:
			timetableObj.addSession(show[0], days[show[0]], show[1], show[2], show[3])
			days[show[0]] += 1

		#Do not change this line
		return timetableObj