"""Tests for trajectory tracing (experimental standalone script)."""
import math
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_dynamics import Car
from utility.visualizer import Visualizer

SCALE = 6.0
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
ROAD_COLOR = [0.4, 0.4, 0.4]


def recreate_tiles(core_env):
    border = [False] * len(core_env.track)
    for i in range(len(core_env.track)):
        good = True
        oneside = 0
        for neg in range(BORDER_MIN_COUNT):
            beta1 = core_env.track[i - neg - 0][1]
            beta2 = core_env.track[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        good &= abs(oneside) == BORDER_MIN_COUNT
        border[i] = good
    for i in range(len(core_env.track)):
        for neg in range(BORDER_MIN_COUNT):
            border[i - neg] |= border[i]

    for i in range(len(core_env.track)):
        alpha1, beta1, x1, y1 = core_env.track[i]
        alpha2, beta2, x2, y2 = core_env.track[i - 1]
        road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
        road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
        road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
        road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
        vertices = [road1_l, road1_r, road2_r, road2_l]
        core_env.fd_tile.shape.vertices = vertices
        t = core_env.world.CreateStaticBody(fixtures=core_env.fd_tile)
        t.userData = t
        c = 0.01 * (i % 3)
        t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
        t.road_visited = False
        t.road_friction = 1.0
        t.fixtures[0].sensor = True
        core_env.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
        core_env.road.append(t)
        if border[i]:
            side = np.sign(beta2 - beta1)
            b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
            b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
            b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
            b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
            core_env.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))


def get_wheel_info(wheel):
    return {"angle": wheel.angle,
            "angular_damping": wheel.angularDamping,
            "angular_velocity": wheel.angularVelocity,
            "brake": wheel.brake,
            "gas": wheel.gas,
            "steer": wheel.steer,
            "inertia": wheel.inertia,
            "linear_damping": wheel.linearDamping,
            "linear_velocity": (wheel.linearVelocity[0], wheel.linearVelocity[1]),
            "omega": wheel.omega,
            "phase": wheel.phase,
            "position": (wheel.position[0], wheel.position[1])}


def set_wheel_info(wheel, wheel_info):
    wheel.angle = wheel_info['angle']
    wheel.angularDamping = wheel_info['angular_damping']
    wheel.angularVelocity = wheel_info['angular_velocity']
    wheel.brake = wheel_info['brake']
    wheel.gas = wheel_info['gas']
    wheel.steer = wheel_info['steer']
    wheel.inertia = wheel_info['inertia']
    wheel.linearDamping = wheel_info['linear_damping']
    wheel.linearVelocity = wheel_info['linear_velocity']
    wheel.omega = wheel_info['omega']
    wheel.phase = wheel_info['phase']
    wheel.position = wheel_info['position']


seed = 2
start_track = 205
env = gym.make('CarRacing-v3', render_mode='human')
env.reset(seed=seed)
core: Any = env.unwrapped  # access internal state via .unwrapped throughout

[env.step(np.array([0, 0, 0], dtype=np.float32)) for _ in range(50)]
env.render()
core.car = Car(core.world, *core.track[start_track][1:4])
vis = Visualizer()

prior_action = [0, 0.2, 0]
STEPS_SO_FAR = 20
HORIZON = 75
action = [0, 1, 0]
plan = [action for _ in range(HORIZON)]
FPS = 50

for _ in range(STEPS_SO_FAR):
    env.step(np.array(prior_action, dtype=np.float32))
    env.render()

car = core.car
start_velocity = (car.hull.linearVelocity[0], car.hull.linearVelocity[1])
start_speed = np.sqrt(np.square(car.hull.linearVelocity[0]) + np.square(car.hull.linearVelocity[1]))
start_position = (car.hull.position[0], car.hull.position[1])
start_angle = car.hull.angle
start_angular_velocity = car.hull.angularVelocity
start_inertia = car.hull.inertia
start_time = core.t
start_reward = core.reward
start_previous_reward = core.prev_reward
start_tile_visited_count = core.tile_visited_count

print(f"\nInitial speed: {start_speed} | velocity: {start_velocity} | position: {start_position}")

current_num_tiles_visited = start_tile_visited_count
total_tile_reward = 0

start_wheels = [get_wheel_info(car.wheels[wheel_num]) for wheel_num in range(4)]
for w in start_wheels:
    print(w)

total_tiles_visited = 0
next_velocity = start_velocity
next_speed = start_speed
next_position = start_position
next_angle = start_angle
next_angular_velocity = start_angular_velocity
next_inertia = start_inertia

for action in plan:
    car.steer(-action[0])
    car.gas(action[1])
    car.brake(action[2])
    car.step(1.0 / FPS)

    core.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
    core.t += 1.0 / FPS
    new_num_tiles = core.tile_visited_count - current_num_tiles_visited
    total_tiles_visited += new_num_tiles
    total_tile_reward += new_num_tiles * (1000.0 / len(core.track))

    core.reward += -0.1 + new_num_tiles * (1000.0 / len(core.track))
    core.prev_reward = core.reward
    current_num_tiles_visited = core.tile_visited_count

    env.render()

    next_velocity = (car.hull.linearVelocity[0], car.hull.linearVelocity[1])
    next_speed = np.sqrt(np.square(next_velocity[0]) + np.square(next_velocity[1]))
    next_position = (car.hull.position[0], car.hull.position[1])
    next_angle = car.hull.angle
    next_angular_velocity = car.hull.angularVelocity
    next_inertia = car.hull.inertia

print(f"\nNext speed: {next_speed} | Next Velocity: {next_velocity} | Next Position: {next_position}")

# Reset env and car back to checkpoint
core.t = start_time
core.reward = start_reward
core.prev_reward = start_previous_reward
core.tile_visited_count = start_tile_visited_count
recreate_tiles(core)

next_wheels = [get_wheel_info(car.wheels[wheel_num]) for wheel_num in range(4)]
for w in next_wheels:
    print(w)

car.destroy()
core.car = Car(world=core.world, init_angle=start_angle, init_x=start_position[0], init_y=start_position[1])

print("RESET WHEELS \n")
for i in range(4):
    set_wheel_info(core.car.wheels[i], start_wheels[i])
reset_wheels = [get_wheel_info(core.car.wheels[wheel_num]) for wheel_num in range(4)]
for w in reset_wheels:
    print(w)
core.car.hull.linearVelocity[0] = start_velocity[0]
core.car.hull.linearVelocity[1] = start_velocity[1]
core.car.hull.angularVelocity = start_angular_velocity

after_velocity = core.car.hull.linearVelocity
after_position = core.car.hull.position
after_speed = np.sqrt(np.square(after_velocity[0]) + np.square(after_velocity[1]))
print(f"\nAfter Reset speed: {after_speed} | Velocity: {after_velocity} | Position: {after_position}")
assert after_speed == start_speed
assert core.tile_visited_count == start_tile_visited_count
assert core.reward == start_reward
assert core.prev_reward == start_previous_reward
assert reset_wheels[0] == start_wheels[0]
assert reset_wheels[1] == start_wheels[1]
assert reset_wheels[2] == start_wheels[2]
assert reset_wheels[3] == start_wheels[3]

env.render()

for action in plan:
    obs, reward, terminated, truncated, _ = env.step(np.array(action, dtype=np.float32))
    env.render()

core.reward = core.reward + total_tiles_visited * 1000.0 / len(core.track)
print(f"\nReal total reward: {core.reward}")
