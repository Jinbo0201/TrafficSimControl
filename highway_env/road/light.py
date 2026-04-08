class TrafficLight:
    RED = 0
    GREEN = 1

    def __init__(self, position, cycle=50):
        self.position = position  # 红绿灯位置 (x, y)
        self.cycle = cycle       # 切换周期
        self.timer = 0
        self.state = self.GREEN  # 初始绿灯

    def step(self):
        # 每一步更新计时，自动切换
        self.timer += 1
        if self.timer >= self.cycle:
            self.state = self.RED if self.state == self.GREEN else self.GREEN
            self.timer = 0

    def is_red(self):
        return self.state == self.RED

    def is_green(self):
        return self.state == self.GREEN