Implementing - AI based - informed vs uninformed algorithms to solve the 15 puzzle problem and compare the runtime.
This problem is based on a 4x4 grid board, and has 16! possible state. Half of those states are unreachable, and yet the problem still has a massive number of states it can reach.
That's exactly where the difference between informed and uninformed algorithms is shown, as can be observed in the output (.png) files.

I've used the Linear Conflict heuristic function, which I found as pseudo code (which I then had to fix) and implement it in my code.
