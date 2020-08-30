import cv2


class PIController:
    def __init__(self, Kp=0.1, Ki=0.002):
        self.Kp = Kp
        self.Ki = Ki
        self.target = 22
        self.integral = 0.0

    def init(self):
        self.integral = 0.0

    def set_target(self, sp):
        self.target = sp

    def update(self, measurement):
        err = self.target - measurement
        self.integral += err
        signal = self.Kp*err + self.Ki*self.integral
        return signal


if __name__ == '__main__':
    controller = PIController()

    while True:
        measurement = 0.0
        signal = controller.update(measurement)
        signal = controller.update(signal)
        print(signal)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break