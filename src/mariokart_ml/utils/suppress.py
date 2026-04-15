import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

class Suppress:
    def __enter__(self):
        # save original file descriptor for later
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)
        self.devnull = open(os.devnull, 'w')
        
        # force new file descriptors to point to /dev/null
        os.dup2(self.devnull.fileno(), self.stdout_fd)
        os.dup2(self.devnull.fileno(), self.stderr_fd)

    def __exit__(self, type, value, traceback):
        # restore original file descriptor
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        
        # cleanup
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        self.devnull.close()