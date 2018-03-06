import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import kuka
import random
import pybullet_data

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class KukaGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps = 1000):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180 
    self._cam_pitch = -40
    self._p = p
    if self._renders:
      cid =-1
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self._seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)
    
    observation_high = np.array([largeValObservation] * observationDim)    



########################################################
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
       action_dim = 3
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None
###########################################################



     gripperEul =  p.getEulerFromQuaternion(gripperOrn)
     #print("gripperEul")
     #print(gripperEul)
     blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos,invGripperOrn,blockPos,blockOrn)
     projectedBlockPos2D =[blockPosInGripper[0],blockPosInGripper[1]]
     blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
     #print("projectedBlockPos2D")
     #print(projectedBlockPos2D)
     #print("blockEulerInGripper")
     #print(blockEulerInGripper)
     
     #we return the relative x,y position and euler angle of block in gripper space
     blockInGripperPosXYEulZ =[blockPosInGripper[0],blockPosInGripper[1],blockEulerInGripper[2]]
     
     #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
     #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
     #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)
     
     self._observation.extend(list(blockInGripperPosXYEulZ))
     return self._observation
  
  def _step(self, action):
    if (self._isDiscrete):
      dv = 0.005
      dx = [0,-dv,dv,0,0,0,0][action]
      dy = [0,0,0,-dv,dv,0,0][action]
      da = [0,0,0,0,0,-0.05,0.05][action]
      f = 0.3
      realAction = [dx,dy,-0.002,da,f]
    
    reward = self._reward()-actionCost
    #print("reward")
    #print(reward)
 
    #print("len=%r" % len(self._observation))
    
    return np.array(self._observation), reward, done, {}

  def _render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]
      
    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter>self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005 
    closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid,maxDist)
     
    if (len(closestPoints)):#(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1
      
      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range (100):
        graspAction = [0,0,0.0001,0,fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle-(0.3/100.)
        if (fingerAngle<0):
          fingerAngle=0
    
      for i in range (1000):
        graspAction = [0,0,0.001,0,fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2]>0.5):
          break

    
      self._observation = self.getExtendedObservation()
      return True
    return False
    
  def _reward(self):
    
    #rewards is height of target object
    blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid,self._kuka.kukaUid,1000, -1, self._kuka.kukaEndEffectorIndex) 

    reward = -1000
    
    numPt = len(closestPoints)
    #print(numPt)
    if (numPt>0):
      #print("reward:")
      reward = -closestPoints[0][8]*10
    if (blockPos[2] >0.2):
      reward = reward+10000
      print("successfully grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
     #print("reward")
    #print(reward)
    return reward

  def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
    return self._reset()
