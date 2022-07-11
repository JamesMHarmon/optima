class Metrics:
    def __init__(self):
        self._metrics = {}

    def update(self, name, value):
        if not name in self._metrics:
            self._metrics[name] = Metric()

        self._metrics[name].update(value)

    def aggregate(self):
        return { name: metric.aggregate() for (name, metric) in self._metrics.items()}

    def reset(self):
        self._metrics = {}

class Metric:
    def __init__(self):
        self.value = 0.0
        self.count = 0

    def update(self, value):
        self.value = self.value + value
        self.count = self.count + 1

    def aggregate(self):
        if self.count == 0:
            return self.value
        return self.value / self.count

    def reset(self):
        self.value = 0.0
        self.count = 0