class TimerContextManager:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.end_time = time.time()
        print(f"Time taken: {self.end_time - self.start_time} seconds")
