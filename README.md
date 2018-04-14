# battleship-reinforcement-learning

Inspired by my colleague, who is playing the battleship game in FB messenger every night, so I come up with an idea to build such an AI program that can boost our wining rate. :)

如果你有微信的话，欢迎关注我的微信公众号： `猫猫的AI游乐园`  会不定期送上有趣的资讯和AI实验互动 :) 
-> 2018.04.13 更新： 给这公众号发任意消息可以与这个battleship AI进行对战 

References:
   1) Simple battleship tensorflow tutoriale http://efavdb.com/battleship/
   2) AlphaZero Algorithm for Gomoku, tensorflow version written by me :), where I reuse the network layer here. https://github.com/Kelvin-Zhong/AlphaZero_Gomoku/blob/master/policy_value_net_tensorflow.py
   
Demo: (5x5 board with two 1x2 ships)

![5x5 board with two 1x2 ships](https://github.com/Kelvin-Zhong/battleship-reinforcement-learning/blob/master/ai_demo.gif)

You can see that the AI learn to hit the dialog line which can maximize its hitting rate.

Develop Enviroment:
* Python 3.6
* Tensorflow 1.7.0

How to train the model ?  

-> `python3 Train.py`

I want to play against the AI ? 

-> `python3 HumanPlay.py`

I want to deploy the AI program to my server so my friends can play against my AI ? 

-> You can take a look at `HumanPlayForServer.py` , it will store the game state into pickle in local directory, so anyone can play the game via different request.

## Game Configurations:

-> Board size: 5 x 5 

-> Ships: A. 1x2, B. 1x2

  You can change whatever you want by changing the `GameConfig.py` file.

## What's your network structure ?
-> I tried two structures:
   1) Two dense connected layers 
   2) 3 layer CNN (similar to the CNN I built for Gomoku on the second reference)
   As a result, I use the latter one, which is slightly better than the first one, though it takes longer time to train.
   You can find and play with the two network structure by simply changing 

## How do you find a good rewarding function ? I know this is the hard
-> I just simply reuse the rewarding function from the first reference. LOL

## Well, how long did you take to train the model ? And what's the performance ?
  1) For the 5x5 board setup, it takes me half an hour for self-playing 10000 games, and the avg #move stablized at 14, compared to 20 at the beginning.
  2) For the 10x10 board setup, still training...

## Additional note/thinking:
-> One of my colleague ask if we can build a good algorithm simply based on DFS but not Deep Learning ? 
    
   Hmm, I still need to think about it, would like to hear you guys' thoughts. (looks like a good interview question for me to ask the candidates)
   
-> The training time GPU/CPU are similar (based on 5x5 board setup), but sometimes CPU will stuck on computation, not sure why, so eventually I train the network on my AWS GPU instance.

