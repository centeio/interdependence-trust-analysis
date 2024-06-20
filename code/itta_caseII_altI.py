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

    def __init__(self,n_task_types):
        super(RobotTrustModel, self).__init__()

        self.n_task_types = n_task_types

        self.can = Parameter(dtype(0.001 * np.ones(self.n_task_types)), requires_grad=True)

        self.will = Parameter(dtype(10 * np.ones(1)), requires_grad=True)


    def forward(self):        
        trust = torch.zeros(self.n_task_types) #create a 1xn_diffs array of 0s

        c = self.sigm(self.can) #convert to [0,1] range
        w = self.sigm(self.will)

        for t_i in range(len(trust)): #calculates current trust vector
            trust[t_i] = self.compute_trust(c, w, t_i)
            #computing the trust estimate for each cell based on the current lower and upper bounds (basically the 3d trust plot)

        if usecuda:
            trust = trust.cuda()
        
        return trust 

    def compute_trust(self, c_vec, w, task_type):
        # calculates trust
        return c_vec[task_type] * w

    
    def sigm(self, x): #sigmoid function to convert [-10,10] (really [-inf,inf]) to [0,1]
        return torch.div(1,torch.add(1,torch.exp(-x)))

num_iter = 10 #run the strategy 10 times

if __name__ == "__main__":

    for iter in range(0, num_iter, 1):

        n_task_types = 3
    
        model = RobotTrustModel(n_task_types) #create a RobotTrustModel object

        if usecuda:
            model.cuda()
        
        learning_rate = 0.001 #how fast you update the l1,u1,l2,u2 parameters is related to this
        weight_decay = 0.0001 

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #create Adam optimizer

        loss_tolerance = 0.0005 #when the loss is below this value, we can stop trying to optimize the lower and upper bounds

        t = 0
        report_period = 200 #how often to print the lower and upper bounds and loss to the screen

        c_prog = [] #array for the progression of capability dimension lower bound
        w_prog = [] #array for the progression of capability dimension upper bound

        tt = [] #array from 0 to some number <= 2199, where the sequence repeats for the number of tasks
        loss_to_save = [] #the mean square error we are minimizing. New value is appended every time Adam is run or passed if loss already within tolerance    
        task_number = [] #the number of the task every time Adam is run or passed if loss is already within tolerance
        task_number_stopping_early = [] #the number of the task when the optimizer is not run becasue loss is already below tolerance or when the loss is not decreasing
        t_count = 0 #value to append to the counter
        counter = [] #giant linspace array [0,1,2,...] for every time Adam is run in total or passed because loss is already within tolerance

        total_obs = np.zeros(n_task_types) #creates nbins x nbins array of 0s for holding the number of tasks that fall into each cell
        total_successes = np.zeros(n_task_types) #creates nbins x nbins array of 0s for the number of human successes in each cell
        obs_probs = np.zeros(n_task_types) #creates nbins x nbins array of 0s for the number of human success probability in each cell

        num_tasks = 500 #Max = 500 based on .mat file. Change to the number of tasks you want to allocate.

        total_num_tasks = [[num_tasks]]      
        perfs = [ [] for _ in range(n_task_types) ] #1 x num_tasks array to hold team successes or failures 

        human_p = [] #array to hold the lambdabar1 requirements for tasks assigned to the human
        robot_p = [] #array to hold the lambdabar1 requirements for tasks assigned to the robot
        tie_p = [] #array to hold the lambdabar1 requirements for tasks that are a tie
        assigned = -1 #to indicate which agent the task is assigned to (-1 means no one yet, 0 means robot, 1 means human)
        human_trust = [ [] for _ in range(n_task_types) ] #array to hold trust in the human for every task
        robot_trust = [ [] for _ in range(n_task_types) ] #array to hold trust in the robot for every task
        human_expected = [] #array to hold expected human reward for every task
        robot_expected = [] #array to hold expected robot reward for every task
        human_perfs = [ [] for _ in range(n_task_types) ] #array to hold human successes or failures for tasks assigned to the human
        human_successes = [ 0 for _ in range(n_task_types) ] #array to hold the task requirements for tasks the human successed on
        human_failures = [ 0 for _ in range(n_task_types) ] #array to hold the task requirements for tasks the human failed on
        robot_perfs = [ [] for _ in range(n_task_types) ] #array to hold robot successes or failures for tasks assigned to the robot
        robot_successes = [ 0 for _ in range(n_task_types) ] #array to hold the task requirements for tasks the robot successed on
        robot_failures = [ 0 for _ in range(n_task_types) ] #array to hold the task requirements for tasks the robot failed on
        human_num_tasks = 0 #the number of tasks assigned to the human
        robot_num_tasks = 0 #the number of tasks assigned to the robot
        tie_num_tasks = 0 #the number of tasks that were initially a tie
    

        total_reward = 0 #the total reward for this task
        max_total_reward = 0 #the max total reward if every task was a success

        #fabricated capability values for the human
    
        human_c = np.array([0.80,0.20,0.90]) #size is n_cap
        human_w = 0.7

        robot_c = np.array([0.69,0.39,0.7])
        robot_w = 1

        #ITTA
        print("Running ATTA Iter " + str(iter))
        for i in range(num_tasks): #iterates from 0 to num_tasks-1 for a total of num_tasks times

            task_type = random.choice(range(n_task_types))
            print("task type", task_type)

            #reward = 0
            #humanCost = 0
            #robotCost = 0

            c_sigm = model.sigm(model.can)
            w_sigm = model.sigm(model.will)

            #compute trust in each agent now based on current belief in lower and upper bounds
            humantrust_i = model.compute_trust(c_sigm, w_sigm, task_type)
            robottrust_i = model.compute_trust(robot_c, robot_w, task_type)

            #append trust for this task to the arrays
            human_trust[task_type] += [humantrust_i]
            robot_trust[task_type] += [robottrust_i]

            #compute the task reward and costs now
            #reward = reward/(2.0) 

            #humanCost = humanCost/(3.0)
            #robotCost = robotCost/(8.0)

            #compute expected total reward for each agent
            #Ehuman = humantrust_i.item()*reward - humanCost
            #Erobot = robottrust_i.item()*reward - robotCost

            #human_expected = np.append(human_expected, Ehuman)
            #robot_expected = np.append(robot_expected, Erobot)

            #assign the task now
            assigned = 1 #to indicate the task is assigned to human

            #observe the task outcome now
            tester = random.random()

            #compute the true trust probability for each agent
            human_outcome_prob = human_c[task_type] * human_w
            robot_outcome_prob = robot_c[task_type] * robot_w

            #print(human_outcome_prob)
            

            perf_i = 0 #performance result
            if assigned == 1: #if the task was assigned to the human
                if tester <= human_outcome_prob:
                    perf_i = 1 # bool result on the ith task
                    human_successes[task_type] += 1
                    #total_reward = total_reward + (reward - humanCost)
                else: #already failure by default on the ith task
                    human_failures[task_type] += 1
                    #total_reward = total_reward - humanCost
                human_perfs[task_type] += [perf_i]
                #max_total_reward = max_total_reward + (reward - humanCost)

                obs_probs[task_type] = human_successes[task_type] / (human_successes[task_type] + human_failures[task_type])
            
            elif assigned == 0:  #if the task was assigned to the robot
                #TODO
                print("should not happen")
                exit()
            
            else: #if assigned == -1 and the task was not assigned to anyone
                raise ValueError("Error: Task not assigned")

            perfs[task_type] += [perf_i] #append the performance to the perfs array
            
            #print("col_i = ", col_i)
            #print("tester = ", tester)
            #print("human_successes array = ", human_successes)
            #print("human_failures array = ", human_failures)
            #print("robot_successes array = ", robot_successes)
            #print("robot_failures array = ", robot_failures)
            #print("human_perfs array = ", human_perfs)
            #print("robot_perfs array = ", robot_perfs)
            #print("perfs array = ", perfs)


            #update trust
            if assigned == 1: #if the task was assigned to the human
                
                ll = torch.mean(torch.pow((model() -  torch.FloatTensor(obs_probs)), 2.0 )) #current loss
                if ll.item() < loss_tolerance: #.item is just the way to extract the value from a torch tensor
                    print("loss is already below tolerance. Not running optimizer.")
                    task_number_stopping_early += [i]

                    task_number += [i]

                    c = model.sigm(model.can) #convert back to correct range [0,1]
                    w = model.will
                    c_prog += [c.detach().numpy()] #get the value out of the tensor and add to the l_1 progression vector
                    w_prog += [w.detach().numpy()]

                    tt += [0]
                    counter = np.append(counter, t_count)
                    t_count = t_count + 1 #increment for the next time

                    loss_to_save += [ll.item()]
                    

                else:
                    t = 0 #we let the optimizer run for a max of 2200 times

                    current_loss = 0 #we will use this to stop the optimization early if the loss has not decreased enough from 200 iterations ago
                    loss_200_iters_ago = 0



                    while t < 2200: #I chose 2200. 1520 was a good number of iterations to converge on the true l1,u1,l2,u2 values for Hebert's simulation, otherwise there are just oscillations around the actual capability star value
 

                        def closure(): #closure function must be defined for pytorch
                            #this will calculate the gradients. this runs everytime.
                            #diff1 = model(bin_c, obs_probs_idxs)
                            
                            #diff = torch.tensor(model(bin_c, obs_probs_idxs) - obs_probs_vect, requires_grad=True)
                            diff = model() -  torch.FloatTensor(obs_probs) #the diff between trust estimated from artificial trust model and trust approximation
                            #print("model diff = ", diff1)
                            #print("obs_probs_vect = ", obs_probs_vect)
                            #print("diff = ", diff)
                            #diff.retain_grad()

                            # loss = torch.tensor(torch.mean( torch.pow(diff, 2.0) ), requires_grad=True) #loss needs to be defined in pytorch
                            
                            loss = torch.mean(torch.pow(diff, 2.0)) #calculate the current loss
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
                        ll = torch.mean( torch.pow( (model() -  torch.FloatTensor(obs_probs)), 2.0 ) )

                        #model(bin_c,obs_probs_idxs) calls the model's forward function
                        #for each bin, it is the trust value as given by the model
                        #find the mean square error after subtracting the obs probabilities

                        c = model.sigm(model.can) #convert back to correct range [0,1]
                        w = model.sigm(model.will)
                        c_prog += [c.detach().numpy()] #get the value out of the tensor and add to the l_1 progression vector
                        w_prog += [w.detach().numpy()]


                        tt += [t]
                        counter = np.append(counter, t_count)

                        loss_to_save += [ll.item()]

                        task_number += [i]

                        if loss_to_save[-1] < loss_tolerance: #if the last loss_to_save value is less than the loss_tolerance
                            #print("\ni = ", task_number[-1]) #print the index of the ith task
                            
                            print("t =", tt[-1]) #print the last tt

                            print("counter = ", counter[-1])

                            print("c =", c) #print the last c value
                            print("w =", w) #print the last w value

                            print("loss", loss_to_save[-1]) #print the last loss_to_save value
                            
                            t_count = t_count + 1

                            break #we are near the end of the optimization so we can stop


                        if t % report_period == 0: #if t is a multiple of report_period = 200
                            #print("\ni = ", task_number[-1]) #print the index of the ith task
                            
                            print("t =", tt[-1]) #print the last tt

                            print("counter = ", counter[-1])

                            print("c =", c) #print the last c value
                            print("w =", w) #print the last w value

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
        res_mat_file_name = "results/carolina/itta_altI_eta50_" + str(iter) + ".mat"
        sio.savemat(res_mat_file_name, res_dict) #save to file   
