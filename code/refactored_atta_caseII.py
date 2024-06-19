# imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.cuda import random
from torch.nn import Parameter

import numpy as np
from numpy.linalg import norm

import scipy.io as sio

import sklearn.metrics as metrics

import pickle

import random

import sys

import matplotlib.pyplot as plt

import math

import time


usecuda = True #start by setting this to True
usecuda = usecuda and torch.cuda.is_available() #will become false if your gpu is not there or not available

dtype = torch.DoubleTensor

#torch.set_grad_enabled(True)

if usecuda: #if the the drivers needed for a GPU are available (from checking above)
    dtype = torch.cuda.FloatTensor #just a different data type to help in the optimization

np.seterr(divide='ignore', invalid='ignore') #to not give warnings/errors when dividing by 0 or NaN



class RobotTrustModel(torch.nn.Module):

    def __init__(self):
        super(RobotTrustModel, self).__init__()

        # number of capabilities
        self.n_cap = 2

        # number of willigness
        # n_will = 2

        #beta is not relevant for artificial trust model (see BTM paper for natural trust)
        self.pre_beta = Parameter(dtype(4.0 * np.ones(self.n_cap)), requires_grad=True)

        #instead of using lower bounds of 0 and upper bounds of 1, the code is more
        #stable when using the range [-10,10] and then converting back later to [0,1].
        #requires_grad=True means it is part of the gradient computation (i.e., the value should be updated when being optimized)
        self.pre_can_l = Parameter(dtype(-10.0 * np.ones(self.n_cap)), requires_grad=True)
        self.pre_can_u = Parameter(dtype( 10.0 * np.ones(self.n_cap)), requires_grad=True)


        # self.pre_will_l = Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True)
        # self.pre_will_u = Parameter(dtype( 10.0 * np.ones(1)), requires_grad=True)


    def forward(self, obs_probs_idxs):
        #obs_probs_idxs are passed in
        
        #n_diffs = obs_probs_idxs.shape[0] #the number of [row,col] indexes (max is nbins x nbins (625))
        trust = torch.zeros(len(obs_probs_idxs)) #create a 1xn_diffs array of 0s

        for j in range(model.n_cap):
            if(self.pre_can_l[j] > self.pre_can_u[j]): #if the lower bound is greater than the upper bound
                buf = self.pre_can_l[j] #switch the l_1 and u_1 values
                self.pre_can_l[j].data = self.pre_can_u[j]
                self.pre_can_u[j].data = buf

        l = self.sigm(self.pre_can_l) #convert to [0,1] range
        u = self.sigm(self.pre_can_u)
        beta = self.pre_beta * self.pre_beta #want beta to be positive to compute trust using the artificial trust model

        i = 0
        for prob_idx_temp in obs_probs_idxs: #loop over the number of the [row,col] indexes (max is nbins x nbins (625))
            # TODO may be optimized
            prob_idxs_vec = np.array(list(prob_idx_temp))
            trust[i] = self.compute_trust(l, u, beta, (prob_idxs_vec+0.5)*(1/nbins))
            i += 1
            #computing the trust estimate for each cell based on the current lower and upper bounds (basically the 3d trust plot)

        if usecuda:
            trust = trust.cuda()
        
        return trust

    def compute_trust(self, l_vec, u_vec, b_vec, p_vec):
        #passing in lower bound capability belief, upper bound capability belief, beta, probability

        trust = 1

        for i in range(self.n_cap):
            l = l_vec[i]
            u = u_vec[i]
            b = b_vec[i]
            p = p_vec[i]

            if b < -50: #this is for natural trust. This never happens for the artificial trust model.
                trust_temp = 1.0 - 1.0 / (b * (u - l)) * torch.log( (1.0 + torch.exp(b * (p - l))) / (1.0 + torch.exp(b * (p - u))) )
                #print("trust temp 1: ", trust_temp)
            
            else: #as long as you pass in a positive beta, we will be calculating artificial trust which doesnt depend on beta
                if p <= l: #if lambdabar is less than the lower bound capability belief
                    trust_temp = torch.tensor([1.0],requires_grad=True) #assign a trust of 1
                    #print("trust temp 2: ", trust_temp)
                elif p > u: #if lambdabar is greater than the upper bound capability belief
                    trust_temp = torch.tensor([0.0],requires_grad=True) #assign a trust of 0
                    #print("trust temp 3: ", trust_temp)
                else:
                    trust_temp = (u - p) / (u - l + 0.0001) #assign trust as a constant slope between u and l. 0.0001 is to not divide by 0.
                    #print("trust temp 4", trust_temp)

            trust = trust * trust_temp

        if usecuda:
            trust = trust.cuda()
        
        return trust #returns the trust in human agent given lower bound l, upper bound u, beta term b, and task requirement lambdabar p 

    def sigm(self, x): #sigmoid function to convert [-10,10] (really [-inf,inf]) to [0,1]
        return torch.div(1,torch.add(1,torch.exp(-x)))

    def sigmoid(self, lambdabar, agent_c):
        #takes in task requirement for one dimension and the agent's actual capability for that dimension
        #calculates true trust to determine the stochastic task outcome

        #if lambdabar == agent_c, the sigmoid output is 0.5
        eta = 1/50.0 #is a good value through testing for good capability updating
        #eta = 1/5.0
        #eta = 1/500.0

        return np.ones(self.n_cap) / (np.ones(self.n_cap) + np.exp((lambdabar - agent_c)/(np.ones(self.n_cap)*eta)))



num_iter = 10 #run the strategy 10 times

if __name__ == "__main__":

    for iter in range(0, num_iter, 1):
    
        model = RobotTrustModel() #create a RobotTrustModel object

        if usecuda:
            model.cuda()

        
        learning_rate = 0.001 #how fast you update the l1,u1,l2,u2 parameters is related to this
        weight_decay = 0.0001 

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #create Adam optimizer

        loss_tolerance = 0.0005 #when the loss is below this value, we can stop trying to optimize the lower and upper bounds

        t = 0
        report_period = 200 #how often to print the lower and upper bounds and loss to the screen

        l_prog = [] #array for the progression of capability dimension lower bound
        u_prog = [] #array for the progression of capability dimension upper bound

        tt = [] #array from 0 to some number <= 2199, where the sequence repeats for the number of tasks
        loss_to_save = [] #the mean square error we are minimizing. New value is appended every time Adam is run or passed if loss already within tolerance    
        task_number = [] #the number of the task every time Adam is run or passed if loss is already within tolerance
        task_number_stopping_early = [] #the number of the task when the optimizer is not run becasue loss is already below tolerance or when the loss is not decreasing
        t_count = 0 #value to append to the counter
        counter = [] #giant linspace array [0,1,2,...] for every time Adam is run in total or passed because loss is already within tolerance

        
        nbins = 10

        total_obs = np.zeros((nbins, nbins)) #creates nbins x nbins array of 0s for holding the number of tasks that fall into each cell
        total_successes = np.zeros((nbins, nbins)) #creates nbins x nbins array of 0s for the number of human successes in each cell
        obs_probs = np.zeros((nbins, nbins)) #creates nbins x nbins array of 0s for the number of human success probability in each cell


        num_tasks = 500 #Max = 500 based on .mat file. Change to the number of tasks you want to allocate.

        total_num_tasks = [[num_tasks]]        
        obs_probs_idxs = set() # set of the row and col pairs for which there is an observed trust approximation [[row col][row col]...]
        obs_probs_vect = [] #the trust approximation corresponding to the row and col pair in obs_probs_idxs
        p = [] #2 x num_tasks array to hold all task requirements
        perfs = [] #1 x num_tasks array to hold team successes or failures 

        human_p = [] #array to hold the lambdabar1 requirements for tasks assigned to the human
        robot_p = [] #array to hold the lambdabar1 requirements for tasks assigned to the robot
        tie_p = [] #array to hold the lambdabar1 requirements for tasks that are a tie
        assigned = -1 #to indicate which agent the task is assigned to (-1 means no one yet, 0 means robot, 1 means human)
        human_trust = [] #array to hold trust in the human for every task
        robot_trust = [] #array to hold trust in the robot for every task
        human_expected = [] #array to hold expected human reward for every task
        robot_expected = [] #array to hold expected robot reward for every task
        human_perfs = [] #array to hold human successes or failures for tasks assigned to the human
        human_successes = [[],[]] #array to hold the task requirements for tasks the human successed on
        human_failures = [[],[]] #array to hold the task requirements for tasks the human failed on
        robot_perfs = [] #array to hold robot successes or failures for tasks assigned to the robot
        robot_successes = [[],[]] #array to hold the task requirements for tasks the robot successed on
        robot_failures = [[],[]] #array to hold the task requirements for tasks the robot failed on
        human_num_tasks = 0 #the number of tasks assigned to the human
        robot_num_tasks = 0 #the number of tasks assigned to the robot
        tie_num_tasks = 0 #the number of tasks that were initially a tie
    

        total_reward = 0 #the total reward for this task
        max_total_reward = 0 #the max total reward if every task was a success

        #fabricated capability values for the human
        #human_l1 = 0.54
        #human_u1 = 0.56
        #human_l2 = 0.74
        #human_u2 = 0.76    
        human_l = np.array([0.54,0.74]) #size is n_cap
        human_u = np.array([0.56,0.76])

        human_beta = np.array([1000,1000]) #remember beta does not matter as long as it is positive

        #fabricated capability values for the robot
        #robot_l1 = 0.69
        #robot_u1 = 0.71
        #robot_l2 = 0.39
        #robot_u2 = 0.41
        robot_l = np.array([0.69,0.39])
        robot_u = np.array([0.71,0.41])

        robot_beta = model.pre_beta * model.pre_beta
        
        human_c = (human_l + human_u)/2.0 #actual capabilities
        robot_c = (robot_l + robot_u)/2.0

        load_file_name = "./results/tasks/TA_normaldist_tasks_" + str(iter) + ".mat"
        fixed_tasks_mat = sio.loadmat(load_file_name) #2x500 tasks
        p_from_mat = fixed_tasks_mat["p"] #an array of 2x500 tasks

        #take the tasks for the number of tasks you want
        p = p_from_mat[0:model.n_cap, 0:num_tasks] #take rows 0 to 1 (not including 2) and num_tasks cols from 0 to num_tasks-1

        #print("p = ", p)
        #print("p[0][0] = ", p[0][0])
        #print("p[1][0] = ", p[1][0])


        #ATTA
        print("Running ATTA Iter " + str(iter))
        for i in range(num_tasks): #iterates from 0 to num_tasks-1 for a total of num_tasks times
            reward = 0
            humanCost = 0
            robotCost = 0

            for j in range(model.n_cap):
                #print("IN LOOP ",model.pre_can_l[j],model.pre_can_u[j])
                if(model.pre_can_l[j] > model.pre_can_u[j]): #if the lower bound is greater than the upper bound
                    buf = model.pre_can_l[j] #switch the l_1 and u_1 values
                    model.pre_can_l[j].data = model.pre_can_u[j]
                    model.pre_can_u[j].data = buf
                    print("l and u switched in capability ", j)

                l_sigm = model.sigm(model.pre_can_l) #convert to [0,1] range
                u_sigm = model.sigm(model.pre_can_u)
                beta = model.pre_beta * model.pre_beta #keep it positive       

                reward = reward + p[j][i]
                humanCost = humanCost + p[j][i]
                robotCost = robotCost + p[j][i]


            #compute trust in each agent now based on current belief in lower and upper bounds
            humantrust_i = model.compute_trust(l_sigm, u_sigm, beta, p[:,i])
            robottrust_i = model.compute_trust(robot_l, robot_u, robot_beta, p[:,i])

            #append trust for this task to the arrays
            human_trust = np.append(human_trust, humantrust_i.item())
            robot_trust = np.append(robot_trust, robottrust_i.item())

            #print("humantrust_i = ", humantrust_i)
            #print("robotrust_i = ", robottrust_i)
            #print("human_trust array = ", human_trust)
            #print("robot_trust array = ", robot_trust)

            if usecuda:
                humantrust_i = humantrust_i.cuda()
                robottrust_i = robottrust_i.cuda()

            #compute the task reward and costs now
            reward = reward/(2.0) 

            humanCost = humanCost/(3.0)
            robotCost = robotCost/(8.0)

            #compute expected total reward for each agent
            Ehuman = humantrust_i.item()*reward - humanCost
            Erobot = robottrust_i.item()*reward - robotCost

            human_expected = np.append(human_expected, Ehuman)
            robot_expected = np.append(robot_expected, Erobot)

            #print("reward = ", reward)
            #print("humanReward = ", humanReward)
            #print("robotReward = ", robotReward)
            #print("humanCost = ", humanCost)
            #print("robotCost = ", robotCost)
            #print("Ehuman = ", Ehuman)
            #print("Erobot = ", Erobot)
            #print("human_expected array = ", human_expected)
            #print("robot_expected array = ", robot_expected)

            #assign the task now
            assigned = -1 #to indicate the task is not assigned yet
            Ediff = abs(Ehuman - Erobot)
            alpha_tolerance = 0.0 #to help balance the distribution of tasks assigned to each agent
            if Ediff <= alpha_tolerance:
                tie_p = np.append(tie_p, p[:,i])
                tie_num_tasks = tie_num_tasks + 1 #the number of tasks that were initially tied
                print("not assigned yet: tie")

                if human_num_tasks <= robot_num_tasks: #if the human has fewer or equal tasks as the robot
                    human_p = np.append(human_p, p[:,i]) #append the task's lambdabar1 requirement
                    assigned = 1
                    human_num_tasks = human_num_tasks + 1
                    print("within tolerance assigned to human")
                else: #the robot has fewer tasks than the human
                    robot_p = np.append(robot_p, p[:,i]) 
                    assigned = 0
                    robot_num_tasks = robot_num_tasks + 1
                    print("within tolerance assigned to robot")
            else:
                if (Ehuman > Erobot):
                    human_p = np.append(human_p, p[:,i]) #append the task's lambdabar1 requirement
                    assigned = 1 #to indicate the task is assigned to the human
                    human_num_tasks = human_num_tasks + 1
                    print("assigned to human")
                    
                elif (Ehuman < Erobot):
                    robot_p = np.append(robot_p, p[:,i]) 
                    assigned = 0 #to indicate the task is assigned to the robot
                    robot_num_tasks = robot_num_tasks + 1
                    print("assigned to robot")

                #the case of Ehuman == Erobot is captured in Ediff <= alpha_tolerance above

            #observe the task outcome now
            col_i = np.vstack(p[:,i]) #col for hstack
            tester = random.random()
            perf_i = 0 #failure by default on the ith task

            #compute the true trust probability for each agent
            human_outcome_prob = np.prod(model.sigmoid(p[:,i], human_c))
            robot_outcome_prob = np.prod(model.sigmoid(p[:,i], robot_c))

            print(human_outcome_prob)

            if assigned == 1: #if the task was assigned to the human

                if tester <= human_outcome_prob:
                    perf_i = 1 #success on the ith task
                    human_successes = np.hstack((human_successes, col_i))
                    total_reward = total_reward + (reward - humanCost)
                else: #already failure by default on the ith task
                    human_failures = np.hstack((human_failures, col_i))
                    total_reward = total_reward - humanCost
                human_perfs = np.append(human_perfs, perf_i)
                max_total_reward = max_total_reward + (reward - humanCost)
            
            elif assigned == 0:  #if the task was assigned to the robot
                
                if tester <= robot_outcome_prob:
                    perf_i = 1
                    robot_successes = np.hstack((robot_successes, col_i))
                    total_reward = total_reward + (reward - robotCost)
                else:
                    robot_failures = np.hstack((robot_failures, col_i))
                    total_reward = total_reward - robotCost
                robot_perfs = np.append(robot_perfs, perf_i)
                max_total_reward = max_total_reward + (reward - robotCost)
            
            else: #if assigned == -1 and the task was not assigned to anyone
                raise ValueError("Error: Task not assigned")

            perfs = np.append(perfs, perf_i) #append the performance to the perfs array
            
            #print("col_i = ", col_i)
            #print("tester = ", tester)
            #print("human_successes array = ", human_successes)
            #print("human_failures array = ", human_failures)
            #print("robot_successes array = ", robot_successes)
            #print("robot_failures array = ", robot_failures)
            #print("human_perfs array = ", human_perfs)
            #print("robot_perfs array = ", robot_perfs)
            #print("perfs array = ", perfs)


            #update the nbins x nbins grid now for approximated trust
            if assigned == 1: #if the task was assigned to the human
                #for cap in range(n_cap): TODO adapt to more capabilities

                j1 = round(p[0][i] * (nbins - 1))
                k1 = round(p[1][i] * (nbins - 1))

                # add indices to set of populated cells in trust matrix
                indices_tupple = (j1, k1)
                obs_probs_idxs.add(indices_tupple)
                indices_list = list(indices_tupple)

                #increment total observation matrix
                total_obs[indices_list] = total_obs[indices_list] + 1 #you have found the cell the task falls in so increase the number of observations for that cell
                if perfs[i] == 1: #if the ith task was a success
                    total_successes[indices_list] = total_successes[indices_list] + 1 #increase the number of successes for that cell

                #update probabilities
                obs_probs[indices_list] = total_successes[indices_list] / total_obs[indices_list] #divide the # of successes in each cell by the # of observations
                obs_probs_vect = []
                for prob_id in obs_probs_idxs: #loop over the number of [row,col] indexes (max is nbins x nbins (625))
                    obs_probs_vect += [obs_probs[prob_id]]
                    #obs_probs_idxs[j,0] is the j th [row,col] row value and obs_probs_idxs[j,1] is the col value
                    #get the observed probability of each cell and store it in obs_probs_vect which is an array of values [x,x,...]

                #obs_probs = dtype(obs_probs) #convert to the right data type
                obs_probs_vect = dtype(obs_probs_vect) #convert to the right data type

                #compute current loss and see if it is less than the loss tolerance.
                #if it is, no need to run the optimizer.
                #else, run the optimizer.
                ll = torch.mean( torch.pow( (model(obs_probs_idxs) - obs_probs_vect), 2.0 ) ) #current loss
                if ll.item() < loss_tolerance: #.item is just the way to extract the value from a torch tensor
                    print("loss is already below tolerance. Not running optimizer.")
                    task_number_stopping_early += [i]

                    task_number += [i]

                    l = model.sigm(model.pre_can_l) #convert back to correct range [0,1]
                    u = model.sigm(model.pre_can_u)
                    l_prog += [l.detach().numpy()] #get the value out of the tensor and add to the l_1 progression vector
                    u_prog += [u.detach().numpy()]

                    tt += [0]
                    counter = np.append(counter, t_count)
                    t_count = t_count + 1 #increment for the next time

                    loss_to_save += [ll.item()]
                    

                else:
                    t = 0 #we let the optimizer run for a max of 2200 times

                    current_loss = 0 #we will use this to stop the optimization early if the loss has not decreased enough from 200 iterations ago
                    loss_200_iters_ago = 0



                    while t < 2200: #I chose 2200. 1520 was a good number of iterations to converge on the true l1,u1,l2,u2 values for Hebert's simulation, otherwise there are just oscillations around the actual capability star value
                    
                        #print("pre_l1 = ", model.pre_l_1.item())
                        #print("pre_u1 = ", model.pre_u_1.item())
                        #print("pre_l2 = ", model.pre_l_2.item())
                        #print("pre_u2 = ", model.pre_u_2.item())

                        #if(model.pre_l_1.item() > model.pre_u_1.item()): #if the lower bound is greater than the upper bound
                        #    buf = model.pre_l_1 #switch the l_1 and u_1 values
                        #    model.pre_l_1 = model.pre_u_1
                        #    model.pre_u_1 = buf
                        #    print("l1 and u1 switched")

                        #if(model.pre_l_2.item() > model.pre_u_2.item()): #if the lower bound is greater than the upper bound
                        #    buf = model.pre_l_2 #switch the l_2 and u_2 values
                        #    model.pre_l_2 = model.pre_u_2
                        #    model.pre_u_2 = buf
                        #    print("l2 and u2 switched")


                        def closure(): #closure function must be defined for pytorch
                            #this will calculate the gradients. this runs everytime.
                            
                            diff = model(obs_probs_idxs) - obs_probs_vect #the diff between trust estimated from artificial trust model and trust approximation
                            #print("model diff = ", diff1)
                            #print("obs_probs_vect = ", obs_probs_vect)
                            #print("diff = ", diff)
                            #diff.retain_grad()

                            # loss = torch.tensor(torch.mean( torch.pow(diff, 2.0) ), requires_grad=True) #loss needs to be defined in pytorch
                            
                            loss = torch.mean( torch.pow(diff, 2.0) ) #calculate the current loss
                            #loss.retain_grad() #Something to try if current implementation doesnt work
                            #loss = torch.mean( torch.pow( (model(bin_c, obs_probs_idxs) - obs_probs_vect), 2.0 ) )
                            #print("loss = ", loss)
                            #print("loss.grad_fn = ", loss.grad_fn)
                            optimizer.zero_grad() #standard command to give. Sets the gradients all to 0.
                            #print("ran zero grad")
                            loss.backward() #take deriv of loss function wrt the model parameters 
                            #pytorch lets you choose a function to minimize. We are minimizing the loss function defined above.
                            #print("ran loss backward")
                            return loss

                        #print("_l1 = ", model.sigm( model.pre_l_1))
                        #print("_u1 = ", model.sigm( model.pre_u_1))
                        #print("_l2 = ", model.sigm( model.pre_l_2))
                        #print("_u2 = ", model.sigm( model.pre_u_2))
                        
                        
                        optimizer.step(closure) #optimizer calculates the gradient and adjusts the parameters that will minimize the loss function
                        #running the optimizer to update the parameters below

                        l = model.sigm(model.pre_can_l) #convert back to correct range [0,1]
                        u = model.sigm(model.pre_can_u)
                        ll = torch.mean( torch.pow( (model(obs_probs_idxs) - obs_probs_vect), 2.0 ) )

                        #model(bin_c,obs_probs_idxs) calls the model's forward function
                        #for each bin, it is the trust value as given by the model
                        #find the mean square error after subtracting the obs probabilities


                        l_prog += [l.detach().numpy()] #get the value out of the tensor and add to the l_1 progression vector
                        u_prog += [u.detach().numpy()]


                        tt += [t]
                        counter = np.append(counter, t_count)

                        loss_to_save += [ll.item()]

                        task_number += [i]

                        if loss_to_save[-1] < loss_tolerance: #if the last loss_to_save value is less than the loss_tolerance
                            print("\ni = ", task_number[-1]) #print the index of the ith task
                            
                            print("t =", tt[-1]) #print the last tt

                            print("counter = ", counter[-1])

                            print("l =", l) #print the last l value
                            print("u =", u) #print the last u value

                            print("loss", loss_to_save[-1]) #print the last loss_to_save value
                            
                            t_count = t_count + 1

                            break #we are near the end of the optimization so we can stop


                        if t % report_period == 0: #if t is a multiple of report_period = 200
                            print("\ni = ", task_number[-1]) #print the index of the ith task
                            
                            print("t =", tt[-1]) #print the last tt

                            print("counter = ", counter[-1])

                            print("l =", l) #print the last l_1 value
                            print("u =", u) #print the last u_1 value

                            print("loss", loss_to_save[-1]) #print the last loss_to_save value
                            
                            if t == 0: #update the current loss and the loss from 200 iterations ago
                                loss_200_iters_ago = -1000
                                current_loss = loss_to_save[-1]
                            else:
                                loss_200_iters_ago = current_loss
                                current_loss = loss_to_save[-1]

                        if abs(current_loss - loss_200_iters_ago) < 0.00000001: #the loss is not decreasing, no point in running optimizer anymore
                            print("loss is not decreasing. Not running optimizer.")
                            #time.sleep(3) #sleep for 3 seconds so I can verify the output in the terminal

                            task_number_stopping_early += [i]

                            t_count = t_count + 1
                            break
                            

                        t_count = t_count + 1
                        t = t + 1 #increment t


        res_dict = {"l": l, "u": u, "tt": tt, "counter": counter, "loss": loss_to_save, "task_number_stopping_early": task_number_stopping_early, "human_p": human_p, "robot_p": robot_p, "human_perfs": human_perfs, "robot_perfs": robot_perfs, "human_successes": human_successes, "human_failures": human_failures, "robot_successes": robot_successes, "robot_failures": robot_failures, "human_num_tasks": human_num_tasks, "robot_num_tasks": robot_num_tasks, "tie_num_tasks": tie_num_tasks, "total_num_tasks": total_num_tasks[0][0], "human_trust": human_trust, "robot_trust": robot_trust, "human_expected": human_expected, "robot_expected": robot_expected, "p": p, "perfs": perfs, "task_number": task_number, "obs_probs": obs_probs, "total_succeses": total_successes , "total_obs": total_obs, "human_l": human_l, "human_u": human_u, "robot_l": robot_l, "robot_u": robot_u, "total_reward": total_reward, "max_total_reward": max_total_reward}
        res_mat_file_name = "results/carolina/refactored_atta_caseII_eta50_" + str(iter) + ".mat"
        sio.savemat(res_mat_file_name, res_dict) #save to file   
