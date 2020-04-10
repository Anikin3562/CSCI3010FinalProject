'''
Code written by Joshua Wry 
Student ID: 100661785
'''

import pygame
import time
import numpy as np
import sys
from datetime import datetime
import random
from scipy.integrate import ode
import matplotlib.pyplot as pl
import time

class doublePendulum(object):

	def __init__(self, r1, r2, m1, m2, a1, a2, tx, ty, name, pend_colour, trace_enabled):
		self.r1 = r1
		self.r2 = r2 
		self.m1 = m1
		self.m2 = m2
		self.a1 = a1
		self.a2 = a2

		# Positions of the bobs
		self.x1 = self.r1 * np.sin(self.a1)
		self.y1 = self.r1 * np.cos(self.a1)

		self.x2 = self.x1 + self.r2 * np.sin(self.a2)
		self.y2 = self.y1 + self.r2 * np.cos(self.a2)

		# Translation values
		self.tx = tx
		self.ty = ty

		# Angular Velocities
		self.a1_v = 0.0
		self.a2_v = 0.0

		# Angular Accelerations
		self.a1_a = 0.0
		self.a2_a = 0.0

		# Radius of bob
		self.radius = 20

		# Acceleration due to gravity
		self.g = 9.81

		# Time step - Linked to FPS
		self.dt = 1/60

		# Holds previous positions of the second bob on the pendulum
		self.previous_positions = []
		self.trace_enabled = trace_enabled
		self.trace_count = 0
		
		# Meters to pixels ratio
		self.pixels_per_meter = 200

		# Object identifier
		self.name = name

		# Colour of the bobs
		self.pend_colour = pend_colour

		# Numerical Solver for integrals
		self.solver = ode(self.f)
		self.solver.set_integrator('dop853')
		self.solver.set_initial_value([self.a1, self.a2, self.a1_v, self.a2_v], t=0)

	# Translate objects with respect to coordinate system
	def translate_x(self, x=0):
		return x + self.tx

	def translate_y(self, y=0):
		return y + self.ty

	# Draws lines which trace second pendulum's position
	def trace_pendulum(self, bg, colour):
		
		for i in range(1, len(self.previous_positions)):			
			pygame.draw.lines(bg, colour, False, [(self.translate_x(int(self.sim_to_screen(self.previous_positions[i-1][0]))), self.translate_y(int(self.sim_to_screen(self.previous_positions[i-1][1])))), (self.translate_x(int(self.sim_to_screen(self.previous_positions[i][0]))), self.translate_y(int(self.sim_to_screen(self.previous_positions[i][1]))))],5)

	def f(self):
		return [self.a1_v, self.a2_v, self.a1_a, self.a2_a]

	def update2(self):

		num1 = -self.g * (2 * self.m1 + self.m2) * np.sin(self.a1)
		num2 = -self.m2 * self.g * np.sin(self.a1 - 2 * self.a2)
		num3 = -2 * np.sin(self.a1 - self.a2) * self.m2
		num4 = self.a2_v * self.a2_v * self.r2 + self.a1_v * self.a1_v * self.r1 * np.cos(self.a1 - self.a2)
		den = self.r1 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * self.a1 - 2 * self.a2))
		self.a1_a = (num1 + num2 + num3 * num4) / den;

		num1 = 2 * np.sin(self.a1 - self.a2)
		num2 = self.a1_v * self.a1_v * self.r1 * (self.m1 + self.m2)
		num3 = self.g * (self.m1 + self.m2) * np.cos(self.a1)
		num4 = self.a2_v * self.a2_v * self.r2 * self.m2 * np.cos(self.a1 - self.a2)
		den = self.r2 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * self.a1 - 2 * self.a2))
		self.a2_a = (num1 * (num2 + num3 + num4)) / den

		if self.solver.successful():
			self.solver.integrate(self.solver.t + self.dt)

		self.a1 = self.solver.y[0]
		self.a2 = self.solver.y[1]
		self.a1_v = self.solver.y[2]
		self.a2_v = self.solver.y[3]

		self.x1 = self.r1 * np.sin(self.a1)
		self.y1 = self.r1 * np.cos(self.a1)

		self.x2 = self.x1 + self.r2 * np.sin(self.a2)
		self.y2 = self.y1 + self.r2 * np.cos(self.a2)

		#print("x1: %f, y1: %f, x2: %f, y2: %f" % (self.x1, self.y1, self.x2, self.y2))
		self.previous_positions.append((self.x2, self.y2))

	def update(self):
		num1 = -self.g * (2 * self.m1 + self.m2) * np.sin(self.a1)
		num2 = -self.m2 * self.g * np.sin(self.a1 - 2 * self.a2)
		num3 = -2 * np.sin(self.a1 - self.a2) * self.m2
		num4 = self.a2_v * self.a2_v * self.r2 + self.a1_v * self.a1_v * self.r1 * np.cos(self.a1 - self.a2)
		den = self.r1 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * self.a1 - 2 * self.a2))
		self.a1_a = (num1 + num2 + num3 * num4) / den;

		num1 = 2 * np.sin(self.a1 - self.a2)
		num2 = self.a1_v * self.a1_v * self.r1 * (self.m1 + self.m2)
		num3 = self.g * (self.m1 + self.m2) * np.cos(self.a1)
		num4 = self.a2_v * self.a2_v * self.r2 * self.m2 * np.cos(self.a1 - self.a2)
		den = self.r2 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * self.a1 - 2 * self.a2))
		self.a2_a = (num1 * (num2 + num3 + num4)) / den


		self.x1 = self.r1 * np.sin(self.a1)
		self.y1 = self.r1 * np.cos(self.a1)

		self.x2 = self.x1 + self.r2 * np.sin(self.a2)
		self.y2 = self.y1 + self.r2 * np.cos(self.a2)

		self.a1_v = self.a1_v + self.dt * self.a1_a
		self.a2_v = self.a2_v + self.dt * self.a2_a 

		self.a1 = self.a1 + self.dt * self.a1_v
		self.a2 = self.a2 + self.dt * self.a2_v
        
		#print("x1: %f, y1: %f, x2: %f, y2: %f" % (self.sim_to_screen(self.x1), self.sim_to_screen(self.y1), self.sim_to_screen(self.x2), self.sim_to_screen(self.y2)))
		self.previous_positions.append((self.x2, self.y2))


	# Provides a conversion for meters to pixels
	def sim_to_screen(self, meters):
		# The function takes meters and outputs pixel value in its place
		return meters * self.pixels_per_meter

class Universe: 

	def __init__(self):
		
		# Ratio of pixels to meter
		self.pixels_per_meter = 200

		# All pendulums to draw
		#self.pendulums = pygame.sprite.Group()
		self.pendulums_dict = {}

		# Numerical method - 0 for Euler's, 1 for RK4
		self.num_method = 0

	def sim_to_screen(self, meters):
		return meters * pixels_per_meter

	# Translate objects with respect to coordinate system
	def translate_x(self, x=0):
		return x + self.tx

	def translate_y(self, y=0):
		return y + self.ty

	def add_pendulum(self, pend):
		self.pendulums_dict[pend.name] = pend

	def set_num_method(self, method):
		if(method == "euler"):
			self.num_method = 0
		else: 
			self.num_method = 1


	def removekey(self, d, key):

		r = dict(d)

		del r[key]

		return r

	def newdict(self, d):
		r = dict(d)
		return r

	def normalize(self, v):
		norm = np.linalg.norm(v)
		if norm == 0:
			return v

		else:
			return v / norm

	# Detects collisions between all objects in universe.
	def detect_collisions(self):
		
		loop_dict = self.newdict(self.pendulums_dict)

		if len(loop_dict) >= 2:
			for o1 in loop_dict:

				obj1 = loop_dict[o1]

				temp_dict = self.removekey(loop_dict, o1)
				
				for o2 in temp_dict:

					obj2 = loop_dict[o2]

					# Get difference in translations to adjust for coordinate system differences
					diff = obj1.tx - obj2.tx

					translate_obj2_x2 = obj2.x2*200 - diff
					translate_obj2_x1 = obj2.x1*200 - diff
					
					# 4 combinations of collisions we have to look for
					distance_x1_x1 = np.abs(translate_obj2_x1 - obj1.x1*200)
					distance_y1_y1 = np.abs(obj2.y1*200 - obj1.y1*200)

					distance_x2_x2 = np.abs(translate_obj2_x2 - obj1.x2*200)
					distance_y2_y2 = np.abs(obj2.y2*200 - obj1.y2*200)

					distance_x1_x2 = np.abs(translate_obj2_x1 - obj1.x2*200)
					distance_y1_y2 = np.abs(obj2.y1*200 - obj1.y2*200)

					distance_x2_x1 = np.abs(translate_obj2_x2 - obj1.x1*200)
					distance_y2_y1 = np.abs(obj2.y2*200 - obj1.y1*200)

					if distance_x2_x2 <= 40 and distance_y2_y2 <= 40:
						#print("collision 1")
						self.collision_response_1(distance_x2_x2, distance_y2_y2, obj1, obj2)


					if distance_x1_x1 <= 40 and distance_y1_y1 <= 40: 
						#print("collision 2")
						self.collision_response_2(distance_x1_x1, distance_y1_y1, obj1, obj2)

					if distance_x1_x2 <= 40 and distance_y1_y2 <= 40: 
						#print("collision 3")
						self.collision_response_3(distance_x1_x2, distance_y1_y2, obj1, obj2)

					if distance_x2_x1 <= 40 and distance_y2_y1 <= 40: 
						#print("collision 4")
						self.collision_response_4(distance_x2_x1, distance_y2_y1, obj1, obj2)


				loop_dict = self.removekey(loop_dict, o1)

	def collision_response_1(self, distance_x, distance_y, obj1, obj2):
		angle = np.arctan2(obj2.y2*200 - obj1.y2*200, obj2.x2*200 - obj1.x2*200)
		distance_between_circles = np.sqrt(distance_x**2 + distance_y**2)
		distance_to_move = (20+20) - distance_between_circles

		obj1.a2 += np.cos(angle) * ( distance_between_circles) / 200
		obj1.a2 += np.sin(angle) * distance_between_circles / 200
		tangent_vector = np.array([obj1.y2 - obj2.y2, -(obj1.x2 - obj2.x2)])
		normalized_tangent_vector = self.normalize(tangent_vector)
		
		radius_of_oscillation = 0.9

		pend1_x_velocity = obj1.a2_v * radius_of_oscillation
		pend1_y_velocity = obj1.a2_v * radius_of_oscillation
		
		pend2_x_velocity = obj2.a2_v * radius_of_oscillation
		pend2_y_velocity = obj2.a2_v * radius_of_oscillation

		relative_velocity = (pend2_x_velocity - pend1_x_velocity, pend2_y_velocity - pend1_y_velocity)

		length = np.dot(relative_velocity, normalized_tangent_vector)

		velocityComponentOnTangent = normalized_tangent_vector * length

		velocityComponentPerpToTangent = relative_velocity - velocityComponentOnTangent

		obj2.a2_v -= velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj2.a2_v -= velocityComponentPerpToTangent[1] / radius_of_oscillation

		obj1.a2_v += velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj1.a2_v += velocityComponentPerpToTangent[1] / radius_of_oscillation
	

	def collision_response_2(self, distance_x, distance_y, obj1, obj2):
		angle = np.arctan2(obj2.y1*200 - obj1.y1*200, obj2.x1*200 - obj1.x1*200)
		distance_between_circles = np.sqrt(distance_x**2 + distance_y**2)
		distance_to_move = (20+20) - distance_between_circles

		obj1.a1 += np.cos(angle) * ( distance_between_circles) / 200
		obj1.a1 += np.sin(angle) * distance_between_circles / 200
		tangent_vector = np.array([obj1.y1 - obj2.y1, -(obj1.x1 - obj2.x1)])
		normalized_tangent_vector = self.normalize(tangent_vector)
		
		radius_of_oscillation = 1.0

		pend1_x_velocity = obj1.a1_v * radius_of_oscillation
		pend1_y_velocity = obj1.a1_v * radius_of_oscillation

		pend2_x_velocity = obj2.a1_v * radius_of_oscillation
		pend2_y_velocity = obj2.a1_v * radius_of_oscillation

		relative_velocity = (pend2_x_velocity - pend1_x_velocity, pend2_y_velocity - pend1_y_velocity)

		length = np.dot(relative_velocity, normalized_tangent_vector)

		velocityComponentOnTangent = normalized_tangent_vector * length

		velocityComponentPerpToTangent = relative_velocity - velocityComponentOnTangent

		obj2.a1_v -= velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj2.a1_v -= velocityComponentPerpToTangent[1] / radius_of_oscillation

		obj1.a1_v += velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj1.a1_v += velocityComponentPerpToTangent[1] / radius_of_oscillation
	
	def collision_response_3(self, distance_x, distance_y, obj1, obj2):
		angle = np.arctan2(obj2.y1*200 - obj1.y1*200, obj2.x2*200 - obj1.x2*200)
		distance_between_circles = np.sqrt(distance_x**2 + distance_y**2)
		distance_to_move = (20+20) - distance_between_circles

		obj1.a1 += np.cos(angle) * ( distance_between_circles) / 200
		obj1.a1 += np.sin(angle) * distance_between_circles / 200
		tangent_vector = np.array([obj1.y1 - obj2.y2, -(obj1.x1 - obj2.x2)])
		normalized_tangent_vector = self.normalize(tangent_vector)
		
		radius_of_oscillation = 0.9

		pend1_x_velocity = obj1.a1_v * radius_of_oscillation
		pend1_y_velocity = obj1.a1_v * radius_of_oscillation

		pend2_x_velocity = obj2.a2_v * radius_of_oscillation
		pend2_y_velocity = obj2.a2_v * radius_of_oscillation
		
		relative_velocity = (pend2_x_velocity - pend1_x_velocity, pend2_y_velocity - pend1_y_velocity)

		length = np.dot(relative_velocity, normalized_tangent_vector)

		velocityComponentOnTangent = normalized_tangent_vector * length

		velocityComponentPerpToTangent = relative_velocity - velocityComponentOnTangent

		obj2.a2_v -= velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj2.a2_v -= velocityComponentPerpToTangent[1] / radius_of_oscillation

		obj1.a1_v += velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj1.a1_v += velocityComponentPerpToTangent[1] / radius_of_oscillation

	def collision_response_4(self, distance_x, distance_y, obj1, obj2):
		
		angle = np.arctan2(obj2.y2*200 - obj1.y1*200, obj2.x2*200 - obj1.x1*200)
		distance_between_circles = np.sqrt(distance_x**2 + distance_y**2)
		distance_to_move = (20+20) - distance_between_circles

		obj1.a1 += np.cos(angle) * ( distance_between_circles) / 200
		obj1.a1 += np.sin(angle) * distance_between_circles / 200
		tangent_vector = np.array([obj1.y1 - obj2.y2, -(obj1.x1 - obj2.x2)])
		normalized_tangent_vector = self.normalize(tangent_vector)
		
		radius_of_oscillation = 0.9

		pend1_x_velocity = obj1.a1_v * radius_of_oscillation
		pend1_y_velocity = obj1.a1_v * radius_of_oscillation
		
		pend2_x_velocity = obj2.a2_v * radius_of_oscillation
		pend2_y_velocity = obj2.a2_v * radius_of_oscillation
		
		relative_velocity = (pend2_x_velocity - pend1_x_velocity, pend2_y_velocity - pend1_y_velocity)

		length = np.dot(relative_velocity, normalized_tangent_vector)

		velocityComponentOnTangent = normalized_tangent_vector * length

		velocityComponentPerpToTangent = relative_velocity - velocityComponentOnTangent

		obj2.a2_v -= velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj2.a2_v -= velocityComponentPerpToTangent[1] / radius_of_oscillation

		obj1.a1_v += velocityComponentPerpToTangent[0] / radius_of_oscillation
		obj1.a1_v += velocityComponentPerpToTangent[1] / radius_of_oscillation

	def draw(self, bg):

		
		for o in self.pendulums_dict:

			obj = self.pendulums_dict[o]

			if(obj.trace_enabled):
				obj.trace_pendulum(bg, obj.pend_colour)
				

			pygame.draw.lines(bg,(0,0,0),False,[(obj.translate_x(),obj.translate_y()),(obj.translate_x(int(obj.sim_to_screen(obj.x1))),obj.translate_y(int(obj.sim_to_screen(obj.y1))))],2)

			pygame.draw.lines(bg,(0,0,0),False,[(obj.translate_x(int(obj.sim_to_screen(obj.x1))),obj.translate_y(int(obj.sim_to_screen(obj.y1)))),(obj.translate_x(obj.sim_to_screen(obj.x2)),obj.translate_y(obj.sim_to_screen(obj.y2)))],2)

			pygame.draw.circle(bg,obj.pend_colour,(obj.translate_x(int(obj.sim_to_screen(obj.x1))),obj.translate_y(int(obj.sim_to_screen(obj.y1)))), 20)

			pygame.draw.circle(bg,obj.pend_colour,(obj.translate_x(int(obj.sim_to_screen(obj.x2))), obj.translate_y(int(obj.sim_to_screen(obj.y2)))), 20)


	def update(self, bg):

		for o in self.pendulums_dict:

			obj = self.pendulums_dict[o] 

			if self.num_method:
				obj.update2()
			
			else:
				obj.update()

		self.draw(bg)


def main():
	pygame.init()

	clock = pygame.time.Clock()
	closed = False

	width, height =(1000,600)
	window = pygame.display.set_mode((width,height))

	universe = Universe()
	# Option 1: Type of simulation (base, chaos, collision)
	# Option 2: Numerical Method: (euler, rk4) 
	# Option 3: Tracing: (true, false)
	sim_type = ["base", "chaos", "collision"]
	numerical_method = ["euler", "rk4"]
	trace_enabled = ["true", "false"]

	if(len(sys.argv) == 4 and sys.argv[1] in sim_type and sys.argv[2] in numerical_method and sys.argv[3] in trace_enabled):
		
		if(sys.argv[3] == "true"):
			trace_flag = True
		
		else: 
			trace_flag = False

		# Base simulation
		if(sys.argv[1] == "base"):
			pend = doublePendulum(1, 0.9, 1, 0.9, np.pi, np.pi/2, width//2, width//6, "pend", (255,0,0), trace_flag)
			universe.add_pendulum(pend)

		# Demonstrate Chaos
		elif(sys.argv[1] == "chaos"):
			pend1 = doublePendulum(1, 0.9, 1, 0.9, np.pi, np.pi/2, width//2, width//6, "pend1", (255,0,0), trace_flag)
			pend2 = doublePendulum(1, 0.9, 1, 0.9, np.pi, np.pi/2+0.00002, width//2, width//6, "pend2", (0,255,0), trace_flag)
			pend3 = doublePendulum(1, 0.9, 1, 0.9, np.pi, np.pi/2+0.00004, width//2, width//6, "pend3", (0,0,255), trace_flag)
			
			universe.add_pendulum(pend1)
			universe.add_pendulum(pend2)
			universe.add_pendulum(pend3)

		# Collisions
		elif(sys.argv[1] == "collision"):

			pend1 = doublePendulum(1, 0.9, 1, 0.9, np.pi, np.pi/2, width//2 + 150, width//6, "pend1", (255,0,0), trace_flag)
			pend2 = doublePendulum(1, 0.9, 1, 0.95, 0, np.pi/2, width//2 - 150, width//6, "pend2", (0,255,0), trace_flag)
			universe.add_pendulum(pend1)
			universe.add_pendulum(pend2)
		
		# Use Euler's
		if(sys.argv[2] == "euler"):
			universe.set_num_method("euler")

		# Use RK4
		else: 
			universe.set_num_method("rk4")

	# Commandline arguments not correctly given.
	else: 
		print("Uses: python test6.py (base, chaos, collision) (euler, rk4) (true, false)")
		sys.exit(0)


	# Width and height of window
	width, height = (1000,600)
	window = pygame.display.set_mode((width,height))

	# Initialize clock
	clock = pygame.time.Clock()
	closed = False


	paused = False
	while not closed: 
		clock.tick(60)

		event = pygame.event.poll()
		
		# Press p to pause 
		if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
			paused = True
			continue

		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit(0)

		if not paused: 

			window.fill((255,255,255))
			universe.update(window)

			if sys.argv[1] == "collision":
				universe.detect_collisions()

			pygame.display.update()
		else:
			# Press r to resume
			if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
				paused = False

	pygame.quit()	

if __name__ == "__main__":
	main()
