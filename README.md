#### TicTacToe Reinforcement Learning...
*this project is not actually the final product*
Yet I'm very excited to share this here because I feel like I have made some progress in this field

##### What is this actually?
This is ttt AI game engine which is entirely `MLP, ANN` it has no intensions of memmory.
the *network* is able to learn whether is right or wrong through Reward, if it wins the game it gets rewarded,
so it learns that it has done a good move and it updates its parameters so that this winning path gets higher probability
again when the network is in the same situation.

### *_Yet the `Network` learns rules also_*
the network is given the freedom to choose its game move when picks the move that is already picked it's given another shot 
to pick another move, And this comes with the cost the network is getting negative reward when it picks the move that's 
elligal. After many times of training it learns not to pick moves that are already picked because it tries to avoid the 
cost.

## This is just MLP,ANN isn't a very deep network that could outsmart a human yet, but scaling this up to even deeper architectures 
## would improve its capabilities...
