# CustomCartPoleEnv.py

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GoLeftEnv(gym.Env):
    """
    CartPole-v1 환경을 Gymnasium API에 맞춰 직접 구현한 클래스.
    공식 문서와 소스 코드를 참고하여 제작되었습니다.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 실제 막대 길이의 절반
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Action Space: 0(왼쪽), 1(오른쪽)
        self.action_space = spaces.Discrete(2)

        # Observation Space: [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.state = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        # 물리 엔진: 카트와 막대의 다음 상태 계산
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)

        # 종료 조건 확인
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not terminated else 0.0
        
        if self.render_mode == "human":
            self.render()

        # Gymnasium API는 5개의 값을 반환해야 함 (obs, reward, terminated, truncated, info)
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 상태를 -0.05 ~ 0.05 사이의 무작위 값으로 초기화
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        
        if self.render_mode == "human":
            self.render()
            
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        carty = 350  # 카트의 Y축 위치
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = self.state
        cartx = x[0] * scale + self.screen_width / 2.0  # 카트의 중앙 X 위치

        # 배경 그리기
        self.screen.fill((255, 255, 255))
        
        # 카트 그리기
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        pygame.draw.polygon(self.screen, (0, 0, 0), cart_coords)

        # 막대 그리기
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole_coords = [(l, b), (l, t), (r, t), (r, b)]
        pole_coords_rotated = []
        for coord in pole_coords:
            coord_x = coord[0] * math.cos(x[2]) - coord[1] * math.sin(x[2])
            coord_y = coord[0] * math.sin(x[2]) + coord[1] * math.cos(x[2])
            pole_coords_rotated.append((coord_x + cartx, carty - coord_y))
        pygame.draw.polygon(self.screen, (204, 153, 102), pole_coords_rotated)
        
        # 막대 축 그리기
        axle_radius = polewidth / 2
        pygame.draw.circle(self.screen, (127, 127, 127), (cartx, carty), axle_radius)
        
        # 바닥 선 그리기
        pygame.draw.line(self.screen, (0, 0, 0), (0, carty), (self.screen_width, carty), 1)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None