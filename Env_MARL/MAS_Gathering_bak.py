import numpy as np
from skimage.transform import resize

# 定义代表游戏中的代理对象（Agent）的类
class AgentObj:
    def __init__(self, coordinates, ty, name, direction=0, mark=0, hidden=0):
        # 初始化代理的位置（x, y），类型（颜色），名称，方向，标记和隐藏状态
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = ty  # 代理的颜色类型，0为红色，1为绿色，2为蓝色
        self.name = name  # 代理的名称
        self.hidden = hidden  # 是否处于隐藏状态，0表示不隐藏
        self.direction = direction  # 代理的方向：0=右，1=上，2=左，3=下
        self.mark = mark  # 标记，用于某些状态的计数

    # 检查代理是否处于隐藏状态
    def is_hidden(self):
        return self.hidden > 0

    # 增加标记并检查是否应该进入隐藏状态
    def add_mark(self, agent_hidden):
        self.mark += 1
        if self.mark >= 2:
            self.mark = 0
            self.hidden = agent_hidden  # 设置为隐藏状态
        return self.mark

    # 减少隐藏时间，直至代理不再隐藏
    def sub_hidden(self):
        if self.hidden > 0:
            self.hidden -= 1
            self.hidden = max(0, self.hidden)  # 确保隐藏状态不为负值
        return self.hidden

    # 将代理向左旋转（顺时针90度）
    def turn_left(self):
        self.direction = (self.direction + 1) % 4
        return self.direction

    # 将代理向右旋转（逆时针90度）
    def turn_right(self):
        self.direction = (self.direction - 1 + 4) % 4
        return self.direction

    # 获取代理向前移动时的坐标增量
    def move_forward_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 1, 0  # 向右
        elif self.direction == 1:
            delta_x, delta_y = 0, -1  # 向上
        elif self.direction == 2:
            delta_x, delta_y = -1, 0  # 向左
        elif self.direction == 3:
            delta_x, delta_y = 0, 1  # 向下
        else:
            assert self.direction in range(4), 'wrong direction'
        return delta_x, delta_y

    # 获取代理向左移动时的坐标增量
    def move_left_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 0, -1  # 向上
        elif self.direction == 1:
            delta_x, delta_y = -1, 0  # 向左
        elif self.direction == 2:
            delta_x, delta_y = 0, 1  # 向下
        elif self.direction == 3:
            delta_x, delta_y = 1, 0  # 向右
        else:
            assert self.direction in range(4), 'wrong direction'
        return delta_x, delta_y

    # 代理向前移动，更新位置
    def move_forward(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_forward_delta()
        self.x = min(max(0, self.x + delta_x), env_x_size - 1)  # 防止越界
        self.y = min(max(0, self.y + delta_y), env_y_size - 1)  # 防止越界
        return self.x, self.y

    # 代理向后移动，类似于move_forward，但方向相反
    def move_backward(self, env_x_size, env_y_size):
        forward_delta_x, forward_delta_y = self.move_forward_delta()
        delta_x, delta_y = -forward_delta_x, -forward_delta_y
        self.x = min(max(0, self.x + delta_x), env_x_size - 1)
        self.y = min(max(0, self.y + delta_y), env_y_size - 1)
        return self.x, self.y

    # 代理向左移动，更新位置
    def move_left(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_left_delta()
        self.x = min(max(0, self.x + delta_x), env_x_size - 1)
        self.y = min(max(0, self.y + delta_y), env_y_size - 1)
        return self.x, self.y

    # 代理向右移动，更新位置
    def move_right(self, env_x_size, env_y_size):
        left_delta_x, left_delta_y = self.move_left_delta()
        delta_x, delta_y = -left_delta_x, -left_delta_y
        self.x = min(max(0, self.x + delta_x), env_x_size - 1)
        self.y = min(max(0, self.y + delta_y), env_y_size - 1)
        return self.x, self.y

    # 代理保持不动
    def stay(self, **kwargs):
        pass

    # 发射光束，根据当前方向返回一系列光束坐标
    def beam(self, env_x_size, env_y_size):
        if self.direction == 0:  # 向右
            beam_set = [(i + 1, self.y) for i in range(self.x, env_x_size - 1)]
        elif self.direction == 1:  # 向上
            beam_set = [(self.x, i - 1) for i in range(self.y, 0, -1)]
        elif self.direction == 2:  # 向左
            beam_set = [(i - 1, self.y) for i in range(self.x, 0, -1)]
        elif self.direction == 3:  # 向下
            beam_set = [(self.x, i + 1) for i in range(self.y, env_y_size - 1)]
        else:
            beam_set = []
        return beam_set

# 定义食物对象
class FoodObj:
    def __init__(self, coordinates, ty=1, hidden=0, reward=1):
        # 初始化食物的位置、类型、隐藏状态和奖励值
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = ty  # 食物类型，影响其颜色
        self.hidden = hidden  # 隐藏状态
        self.reward = reward  # 被吃掉时的奖励值

    # 检查食物是否隐藏
    def is_hidden(self):
        return self.hidden > 0

    # 当食物被吃掉时，食物进入隐藏状态并返回奖励
    def eat(self, food_hidden):
        self.hidden = food_hidden
        return self.reward

    # 减少食物的隐藏时间
    def sub_hidden(self):
        if self.hidden > 0:
            self.hidden -= 1
            self.hidden = max(0, self.hidden)
        return self.hidden

# 定义游戏环境类
class GameEnv:
    def __init__(self, width=41, height=21, agent_hidden=20, food_hidden=15):
        # 初始化游戏环境的宽度、高度、代理和食物的隐藏时间等
        self.food_objects = None
        self.agent1_beam_set = None
        self.agent1_actions = None
        self.agent2_actions = None
        self.agent2 = None
        self.agent1 = None
        self.agent2_beam_set = None
        self.size_x = width
        self.size_y = height
        self.agent_hidden = agent_hidden  # 代理隐藏持续时间，延长至20步
        self.food_hidden = food_hidden    # 食物隐藏持续时间，延长至15步
        self.reset()  # 初始化环境

    # 重置游戏环境，创建代理和食物对象
    def reset(self):
        # 创建两个代理对象，初始化位置和方向
        self.agent1 = AgentObj(coordinates=(0, self.size_y // 2), ty=2, name='agent1')
        self.agent2 = AgentObj(coordinates=(self.size_x - 1, self.size_y // 2), ty=0, name='agent2', direction=2)
        # 定义代理的可执行动作集合
        self.agent1_actions = [self.agent1.move_forward, self.agent1.move_backward, self.agent1.move_left,
                               self.agent1.move_right, self.agent1.turn_left, self.agent1.turn_right,
                               self.agent1.beam, self.agent1.stay]
        self.agent2_actions = [self.agent2.move_forward, self.agent2.move_backward, self.agent2.move_left,
                               self.agent2.move_right, self.agent2.turn_left, self.agent2.turn_right,
                               self.agent2.beam, self.agent2.stay]
        # 初始化光束轨迹为空
        self.agent1_beam_set = []
        self.agent2_beam_set = []
        # 初始化食物对象，放置在预设的区域
        self.food_objects = []
        center_x = self.size_x // 2
        center_y = self.size_y // 2
        for x in range(center_x - 5, center_x + 6):
            delta = x - (center_x - 5) if x - (center_x - 5) < (center_x + 5) - x else (center_x + 5) - x
            self.food_objects.append(FoodObj(coordinates=(x, center_y)))
            for i in range(1, delta + 1):
                self.food_objects.append(FoodObj(coordinates=(x, center_y - i)))
                self.food_objects.append(FoodObj(coordinates=(x, center_y + i)))

    # 执行两个代理的动作并处理相互作用
    def move(self, agent1_action, agent2_action):
        # 检查传入的动作是否有效
        assert agent1_action in range(8), 'agent1 take wrong action'
        assert agent2_action in range(8), 'agent2 take wrong action'

        # 记录代理的旧位置
        agent1_old_x, agent1_old_y = self.agent1.x, self.agent1.y
        agent2_old_x, agent2_old_y = self.agent2.x, self.agent2.y

        # 更新代理的隐藏状态
        self.agent1.sub_hidden()
        self.agent2.sub_hidden()

        # 重置光束轨迹
        self.agent1_beam_set = []
        self.agent2_beam_set = []

        # 为每个动作传递正确的参数
        if not self.agent1.is_hidden():
            if agent1_action in [0, 1, 2, 3]:  # 移动类动作
                self.agent1_actions[agent1_action](env_x_size=self.size_x, env_y_size=self.size_y)
            elif agent1_action == 6:  # 发射光束
                self.agent1_beam_set = self.agent1_actions[agent1_action](env_x_size=self.size_x, env_y_size=self.size_y)
            else:  # 旋转或停留类动作
                self.agent1_actions[agent1_action]()
        else:
            self.agent1_beam_set = []

        if not self.agent2.is_hidden():
            if agent2_action in [0, 1, 2, 3]:  # 移动类动作
                self.agent2_actions[agent2_action](env_x_size=self.size_x, env_y_size=self.size_y)
            elif agent2_action == 6:  # 发射光束
                self.agent2_beam_set = self.agent2_actions[agent2_action](env_x_size=self.size_x, env_y_size=self.size_y)
            else:  # 旋转或停留类动作
                self.agent2_actions[agent2_action]()
        else:
            self.agent2_beam_set = []

        # 检查代理之间是否发生碰撞，若发生则恢复原位置
        if not self.agent1.is_hidden() and not self.agent2.is_hidden() and \
                ((self.agent1.x == self.agent2.x and self.agent1.y == self.agent2.y) or
                 (self.agent1.x == agent2_old_x and self.agent1.y == agent2_old_y and
                  self.agent2.x == agent1_old_x and self.agent2.y == agent1_old_y)):
            self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y

        # 处理食物和光束的相互作用，计算奖励
        agent1_reward = 0
        agent2_reward = 0
        for food in self.food_objects:
            food.sub_hidden()  # 更新食物的隐藏状态
            if not food.is_hidden():
                if not self.agent1.is_hidden() and food.x == self.agent1.x and food.y == self.agent1.y:
                    agent1_reward += food.eat(self.food_hidden)
                elif not self.agent2.is_hidden() and food.x == self.agent2.x and food.y == self.agent2.y:
                    agent2_reward += food.eat(self.food_hidden)

        # 如果代理被另一方的光束命中，则增加标记并可能进入隐藏状态
        if not self.agent1.is_hidden() and (self.agent1.x, self.agent1.y) in self.agent2_beam_set:
            self.agent1.add_mark(self.agent_hidden)
        if not self.agent2.is_hidden() and (self.agent2.x, self.agent2.y) in self.agent1_beam_set:
            self.agent2.add_mark(self.agent_hidden)

        return agent1_reward, agent2_reward

    # 生成当前环境的矩阵表示（视觉化）
    def contribute_metrix(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3])  # 初始化一个矩阵，表示RGB颜色
        a[1:-1, 1:-1, :] = 0  # 设置边界内的区域为黑色

        # 将光束轨迹绘制到矩阵上
        for x, y in self.agent1_beam_set:
            a[y + 1, x + 1, :] = [1, 1, 0.7]  # 淡黄色光束
        for x, y in self.agent2_beam_set:
            a[y + 1, x + 1, :] = [1, 1, 0.7]

        # 将食物绘制到矩阵上
        for food in self.food_objects:
            if not food.is_hidden():
                a[food.y + 1, food.x + 1, :] = [1 if i == food.type else 0 for i in range(3)]  # 按食物类型设置颜色

        # 将代理绘制到矩阵上
        for i in range(3):
            if not self.agent1.is_hidden():
                delta_x, delta_y = self.agent1.move_forward_delta()  # 显示代理的方向
                if 0 <= self.agent1.x + delta_x < self.size_x and 0 <= self.agent1.y + delta_y < self.size_y:
                    a[self.agent1.y + 1 + delta_y, self.agent1.x + 1 + delta_x, i] = 0.5  # 前方为半透明
                a[self.agent1.y + 1, self.agent1.x + 1, i] = 1 if i == self.agent1.type else 0  # 当前格为实体
            if not self.agent2.is_hidden():
                delta_x, delta_y = self.agent2.move_forward_delta()
                if 0 <= self.agent2.x + delta_x < self.size_x and 0 <= self.agent2.y + delta_y < self.size_y:
                    a[self.agent2.y + 1 + delta_y, self.agent2.x + 1 + delta_x, i] = 0.5
                a[self.agent2.y + 1, self.agent2.x + 1, i] = 1 if i == self.agent2.type else 0

        return a

    # 实现 get_state 函数，将环境矩阵展平成一维向量
    def get_state(self):
        a = self.contribute_metrix()  # 获取环境状态矩阵
        return a.flatten()  # 展平成一维数组以供 DQN 使用

    # 渲染环境，用于游戏进行时的显示
    def render_env(self):
        a = self.contribute_metrix()  # 获取当前矩阵表示

        # 调整矩阵大小以便更清晰显示
        b = resize(a[:, :, 0], (5 * self.size_y, 5 * self.size_x), order=0, preserve_range=True, anti_aliasing=False)
        c = resize(a[:, :, 1], (5 * self.size_y, 5 * self.size_x), order=0, preserve_range=True, anti_aliasing=False)
        d = resize(a[:, :, 2], (5 * self.size_y, 5 * self.size_x), order=0, preserve_range=True, anti_aliasing=False)

        a = np.stack([b, c, d], axis=2)  # 组合为一个RGB图像
        return a

    # 用于训练时的渲染，将矩阵大小调整为标准尺寸
    def train_render(self):
        a = self.contribute_metrix()
        b = resize(a[:, :, 0], (84, 84), order=0, preserve_range=True, anti_aliasing=False)
        c = resize(a[:, :, 1], (84, 84), order=0, preserve_range=True, anti_aliasing=False)
        d = resize(a[:, :, 2], (84, 84), order=0, preserve_range=True, anti_aliasing=False)

        a = np.stack([b, c, d], axis=2)
        return a

    # 进行一步动作交互
    def step(self, agent1_action, agent2_action):
        # 执行动作，获取奖励
        r1, r2 = self.move(agent1_action, agent2_action)
        next_state = self.get_state()  # 获取更新后的状态

        done = False
        # 检查代理是否越界（理论上不会发生，因为移动函数中已处理）
        if (self.agent1.x < 0 or self.agent1.x >= self.size_x or
                self.agent1.y < 0 or self.agent1.y >= self.size_y):
            done = True  # agent1 移动到了边界外
        elif (self.agent2.x < 0 or self.agent2.x >= self.size_x or
              self.agent2.y < 0 or self.agent2.y >= self.size_y):
            done = True  # agent2 移动到了边界外

        # 返回一个四元组：新的状态、奖励、是否结束、以及可选信息
        return next_state, r1, done, {"agent2_reward": r2}