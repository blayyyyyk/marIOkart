import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

class Suppress:
    def __enter__(self):
        # 1. Save the original file descriptors so we can restore them later
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)
        
        # 2. Open /dev/null
        self.devnull = open(os.devnull, 'w')
        
        # 3. Force the system FDs (1 and 2) to point to /dev/null
        os.dup2(self.devnull.fileno(), self.stdout_fd)
        os.dup2(self.devnull.fileno(), self.stderr_fd)

    def __exit__(self, type, value, traceback):
        # 4. Restore the original FDs
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        
        # 5. Clean up
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        self.devnull.close()