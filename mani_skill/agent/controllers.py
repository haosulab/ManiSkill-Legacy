import numpy as np


class LPFilter:
    def __init__(self, control_freq, cutoff_freq):
        dt = 1 / control_freq
        wc = cutoff_freq * 2 * np.pi
        y_cos = 1 - np.cos(wc * dt)
        self.alpha = -y_cos + np.sqrt(y_cos ** 2 + 2 * y_cos)
        self.y = 0

    def next(self, x):
        self.y = self.y + self.alpha * (x - self.y)
        return self.y


class PIDController:
    def __init__(self, kp, ki, kd, control_freq, output_range):
        """
        Args:
            kp: PID Kp term
            ki: PID Ki term
            kd: PID Kd term
            control_freq: control frequency in Hz
            output_range: [low, high] for output signal
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = 1 / control_freq
        self.output_range = output_range
        self.reset()

    def reset(self):
        self._prev_err = None
        self._cum_err = 0

    def control(self, current, target):
        err = target - current
        if self._prev_err is None:
            self._prev_err = err

        value = (
            self.kp * err
            + self.kd * (err - self._prev_err) / self.dt
            + self.ki * self._cum_err
        )

        self._prev_err = err
        self._cum_err += self.dt * err

        return np.clip(value, self.output_range[0], self.output_range[1])


class VelocityController:
    def __init__(self, lp_filter):
        """
        Args:
            lp_filter: None|LPFilter for filtering output velocity
        """
        self.lp_filter = lp_filter

    def control(self, current, target):
        """
        Args:
            current: current velocity
            target: target velocity
        Returns:
            filtered target velocity for low level PD controller
        """
        if self.lp_filter is None:
            return target
        return self.lp_filter.next(target)


class PositionController:
    def __init__(self, velocity_pid, lp_filter):
        """
        Args:
            velocity_pid: PIDController for converting position signal to velocity signal
            lp_filter: None|LPFilter for filtering output velocity
        """
        self.velocity_pid = velocity_pid
        self.lp_filter = lp_filter

    def control(self, current, target):
        """
        Args:
            current: current position
            target: target position
        Returns:
            target velocity for low level PD controller
        """
        target_vel = self.velocity_pid.control(current, target)
        if self.lp_filter is not None:
            target_vel = self.lp_filter.next(target_vel)
        return target_vel


def test():
    input_time = np.linspace(0, 4, 400)  # 100 hz sampling rate
    input_signal = np.sin(input_time * 40 * np.pi * 2) + np.sin(
        input_time * 20 * np.pi * 2
    )  # 40 hz signal + 20 hz signal

    filter1 = LPFilter(100, 40)
    filter2 = LPFilter(100, 10)

    y1 = [filter1.next(x) for x in input_signal]
    y2 = [filter2.next(x) for x in input_signal]

    import matplotlib.pyplot as plt

    plt.subplot(3, 1, 1)
    plt.plot(input_time, input_signal)
    plt.subplot(3, 1, 2)
    plt.plot(input_time, y1)
    plt.subplot(3, 1, 3)
    plt.plot(input_time, y2)
    plt.show()


