# so101-behaviour-cloning-from-scratch

Behaviour cloning for an SO-101 robot arm that picks up one cube and stacks it on another, in
simulation. The policies are written from scratch, without imitation-learning libraries, and
trained on demonstrations from a scripted expert:

1. MLP
2. Action-chunking transformer (ACT)
3. Flow-matching policy

All three start from low-dimensional state; camera input comes after that.
