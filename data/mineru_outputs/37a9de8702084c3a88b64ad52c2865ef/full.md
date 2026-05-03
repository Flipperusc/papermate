# Game-RL: Synthesizing Multimodal Verifiable Game Data to Boost VLMs' General Reasoning

Jingqi Tong $^{1,2,*}$ , Jixin Tang $^{1,*}$ , Hangcheng Li $^{1,*}$ , Yurong Mou $^{1,*}$ Ming Zhang $^{1}$ , Jun Zhao $^{1,\dagger}$ , Yanbo Wen $^{1}$ , Fan Song $^{1}$ , Jiahao Zhan $^{1}$ Yuyang Lu $^{1}$ , Chaoran Tao $^{1}$ , Zhiyuan Guo $^{1}$ , Jizhou Yu $^{1}$ , Tianhao Cheng $^{1}$ , Zhiheng Xi $^{1}$ , Changhao Jiang $^{1}$ , Zhangyue Yin $^{1}$ , Yining Zheng $^{1}$ , Weifeng Ge $^{1}$ , Guanhua Chen $^{3}$ , Tao Gui $^{1,2}$ , Xipeng Qiu $^{1,2,\dagger}$ , Qi Zhang $^{1\dagger}$ , Xuanjing Huang $^{1}$

$^{1}$ Fudan University, $^{2}$ Shanghai Innovation Institute, $^{3}$ SUSTech

*Equal contribution, †Corresponding authors

# Abstract

Vision-language reinforcement learning (RL) has primarily focused on narrow domains (e.g. geometry or chart reasoning). This leaves broader training scenarios and resources underexplored, limiting the exploration and learning of Vision Language Models (VLMs) through RL. We find video games inherently provide rich visual elements and mechanics that are easy to verify. To fully use the multimodal and verifiable reward in video games, we propose Game-RL, constructing diverse game tasks for RL training to boost VLMs general reasoning ability. To obtain training data, we propose Code2Logic, a novel approach that adapts game code to synthesize game reasoning task data, thus obtaining the GameQA dataset of 30 games and 158 tasks with controllable difficulty gradation. Unexpectedly, RL training solely on GameQA enables multiple VLMs to achieve performance improvements across 7 diverse vision-language benchmarks, demonstrating the value of Game-RL for enhancing VLMs' general reasoning. Furthermore, this suggests that video games may serve as valuable scenarios and resources to boost general reasoning abilities.

Date: May 20, 2025

Correspondence: Jingqi Tong at jqtong25@m.fudan.edu.cn, Jun Zhao at zhaoj19@fudan.edu.cn, Xipeng Qiu at xpqiu@fudan.edu.cn, Qi Zhang at qz@fudan.edu.cn

Repository: https://github.com/tongjingqi/Game-RL

# 1 Introduction

Vision language models have achieved impressive progress in basic tasks such as image description and vision question answering. However, they still struggle with diverse and complex tasks that require multi-step reasoning in real-world scenarios [18, 33]. One key reason is that vision-language RL primarily focused on narrow domains, mainly including geometry [22, 35] and chart reasoning [8], as shown in Table 1. This leaves broader training scenarios underexplored, limiting the exploration and learning of VLMs through RL [10, 14, 18, 36].

We recognize that video games inherently have three advantages: First, they can provide rich visual elements and scenes with texts. Second, their mechanics are easy to verify, so we can synthesize reliable tasks with verifiable results. Third, their environments are fully controllable and easy to modify, providing convenience

Table 1 Our GameQA dataset extends RL training scenarios for VLMs to the domain of video games, providing diverse verifiable game tasks along with controllable difficulty. Size means sum of the number of VQA pairs in train set and test set. Ration. Annot. means has annotated reasoning process. 3D Scene means some games of it are 3D scene. Detailed introduction can be found in Section 6.   

<table><tr><td>Reasoning Domain</td><td>Related Work</td><td>Train Set</td><td>Game Count</td><td>Ration. Annot.</td><td>3D Scene</td><td>Adjustable Difficulty</td></tr><tr><td></td><td>MultiMath (Peng et al. [22])</td><td>✓</td><td>N/A</td><td>✓</td><td>×</td><td>×</td></tr><tr><td></td><td>Geo170k (Gao et al. [6])</td><td>✓</td><td>N/A</td><td>✓</td><td>×</td><td>✓</td></tr><tr><td></td><td>MathV360K (Jiang et al. [11])</td><td>✓</td><td>N/A</td><td>✓</td><td>×</td><td>✓</td></tr><tr><td rowspan="5">Game</td><td>ING-VP (Zhang et al. [33])</td><td>×</td><td>6</td><td>×</td><td>×</td><td>✓</td></tr><tr><td>BALROG (Paglieri et al. [21])</td><td>×</td><td>6</td><td>×</td><td>×</td><td>×</td></tr><tr><td>VideoGameBench (Zhang et al. [32])</td><td>×</td><td>23</td><td>×</td><td>✓</td><td>×</td></tr><tr><td>VCbench (Li et al. [13])</td><td>×</td><td>10</td><td>×</td><td>×</td><td>✓</td></tr><tr><td>GameQA (Ours)</td><td>✓</td><td>30</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

for controlling the difficulty. Although existing works found game is a good arena to evaluate VLMs reasoning ability [13, 21, 32, 33], they did not apply training on it. One possible reason is they did not transform game data into Visual Question and Answer (VQA) format to adapt to training, as shown in Table 1.

To fully use the multimodal and verifiable rewards in game, we propose Game-RL, constructing game scenarios for RL training to boost VLMs general reasoning ability. Then we propose Code2Logic, a novel approach adapts game code to synthesize game reasoning task data, as shown in Figure 1. Code2Logic establishes mapping from game code to reasoning logic. Using Code2Logic, we have constructed the GameQA, which is a Visual question Answer (VQA) dataset to train and evaluate the reasoning capabilities of VLMs. GameQA includes 30 games, 158 tasks and 140K questions with controllable difficulty gradation, covering 4 cognitive categories, as shown in Figure 2 and Figure 3. Unexpectedly, despite training solely on game tasks, using Group Relative Policy Optimization (GRPO) [7], several SOTA open-source VLMs demonstrated out of domain generalization, specifically Qwen2.5-VL-7B achieving $2.33\%$ improvement across 7 diverse vision-language benchmarks.

Our main contributions are as follows:

- We propose Game-RL, constructing game tasks for RL training to boost VLMs general reasoning ability. To obtain training data, We propose Code2Logic, a novel approach that adapts game code to synthesize diverse game reasoning task data.   
- Using Code2Logic, we developed the GameQA dataset, which includes 30 games, 158 tasks and 140K questions with controllable difficulty gradation.   
- Our experiments show that multiple VLMs achieve performance improvements across 7 diverse vision-language benchmarks through RL training solely on GameQA, demonstrating the value of Game-RL for enhancing VLMs' general reasoning. This also suggests that video games may serve as valuable scenarios and resources to boost general reasoning abilities.

# 2 Code2Logic: Synthesizing Multimodal Verifiable Game Data

We propose Code2Logic to synthesize verifiable game reasoning tasks data via adapting game code. Code2Logic comprises three core steps: game code construction, task template design, and data engine construction for dataset generation, as shown in Figure 1. Code2Logic establishes a mapping from game code to reasoning logic.

# Step 1: Game Code Construction

Prompt: Create an interactive Sokoban game.

class Sokoban: def __init__(self, x, y, ...) : self(player = (x, y) selfboxes $=$ ... self/goals $=$ ... self.walls $=$ ... def move(self, direction): new_x,new_y $=$ .. if(new_x,new_y) in self.walls: return self.cant_move...) if(new_x,new_y) in selfboxes: return self.push_box(...) return self.move_player(...

![](images/09c1ede941926cc2ff8d4877628ae15efb9e9eece185cacf17670a362bcd2175.jpg)

# Final Result: Data Samples Generated by Data Engine

# Question:

This is a Sokoban puzzle ... If player moves: Left, Up, Down, Down, where will player end up? Options: [1] (1, 1) [2] (1, 2) [3] (1, 3) [4] (2, 1) [5] (2, 2) [6] (2, 3) [7] (3, 1) [8] (3, 2)

# Analysis:

Player position: (2,3)

Move 1 - Left: Player moves from (2,3) to (2,2).

Move 2-Up: Player moves from (2,2) to (1,2).

Final position: (2, 2). Choose option 5.

![](images/f2e09407b0bbbebe7fa4a8bb511b2b99cefe4c64830a37dbee5898fd4bac3f15.jpg)  
Figure 1 Overview of Code2Logic approach. The process involves three main steps: (1) using LLMs to construct game code. (2) LLM-assisted design of the task templates including question and analysis templates based on the generated game code. Each task template condenses one type of reasoning pattern in the game. (3) Using LLMs to construct a data engine that directly reuses the core game code from the first step, including functions like move. (4) After these main steps, the data engine is executed to fill in the task templates developed in Step 2 and generate data samples, as illustrated in the Final Result Section.

# Step 2: Task Template Design

Prompt: Refine man-made Task Template.

Sokoban code: ...

# Question Template:

This is a Sokoban puzzle ... If player moves:

<move_seq>, where will player end up?

Options: <option_1> ... <option_8>

# Analysis Template:

Player position: <pos>

Move 1 - <dir>: Player moves from <pos> to <pos>.

Move 2 -<dir>: Player moves from <pos> to <pos>.

\*\*\*

Final position: <pos>. Choose option <number>.

# Step 3: Data Engine Construction

Prompt: Write a program that executes Sokoban code to fill in Task template, and output the dataset. Sokoban code: ... Task template: ...

class DataEngine:

def generate_sokoban_sample(self):

sokoban $=$ Sokoban(..）

moves $=$ self_generate_randommoves()

state_record $=$ [sokoban.move (move) for move in moves]

return QA_template.format(state_record)

# 2.1 Game code construction

The first step is to construct the code of the target video game. Game code defines the state space and contains some core functions encoding the transition rules of the game. These functions can be reused and adapted for data engine construction.

Taking Sokoban as an example, it is simple enough to be easily built with one-line prompt using LLMs such as Claude 3.5 and GPT-4o. Sokoban game state is only composed of wall, player, box and target, while action only includes moving in four directions. The prompt used for game code generation and pseudocode of the game is illustrated in Figure 1 (Step 1). Meanwhile, the Sokoban move function within Sokoban code can inspire the "State Prediction" questions, such as "Predict where will player end up after these steps", and this function can be reused for data engine construction.

# 2.2 Game task and QA Template design

The second step is to design game task and question-answer templates (QA Templates) according to each video game. Game task design base on the visual element and action space of video game generated in the first step. Taking Sokoban as an example, we can design a task about predicting the position of player after moves according to a game state screen, so we can propose a question and an answer. The question is "If the player moves left, up, down, down, where is the player?" and the answer is "The position of player is (2,2)." Then we can generalize these questions and answers into a QA Template, as shown in figure 1 (Step 2). Meanwhile, we can fill contents into a QA Template to get many instances. Detailed QA Templates of Sokoban is provided in Appendix G.1.

In GameQA dataset, each task is defined as a group of samples from the same game that share the same QA template. We categorize the reasoning tasks into three types, including "Target Perception Task", "State

# 3D Reconstruction

![](images/38c29b3da3a66590a38b8eb65e666ecf333fc95426089b4048fabfc4026ff104.jpg)  
3D Voxel Reconstruction Game

Introduction: Add cubes (voxels) until the structure's front, side, and top views match the target projections. Every new cube must touch an existing cube face-to-face.

[Easy] Q2 (Voxel position identification):

<Question> Which of the positions contains a voxel?

<Analysis> Analyze each option: ...

<Answer>Option 2.(1,1,1).

[Hard]Q6(Min voxel addition requirement) :

<Question> What is the minimum number of

voxels needed to make it match both target projections?

<Analysis>

1. Analysis of Current Information: ...

2. Analysis of Y-Z Projection (Front View): ...

3. Analysis of X-Z Projection (Side View): ...

4. Finding Required Positions:

5. Connectivity Analysis and Completion: ...

6. Verifying Optimality:

Therefore, the minimum number of voxels

needed to complete the reconstruction is 4.

<Answer> 4.

Game Category: 3D Spatial Perception and Understanding

# Tangram

![](images/188b972b24234f96430e78d50a0ab244a3b6f9cfaa4b3f50d251609c9d016dbc.jpg)

![](images/cc7154deaedbd2d9d05d325343fd0bb40476e213766fbc5c971a54c1650ff7c5.jpg)

Introduction: Fit the pieces back onto the main board to complete the shape. Pieces must touch at least one edge, not just a corner. Some pieces are shown below the board. Grids with same color belong to the same piece.

Game Category: Pattern Recognition and Matching

[Easy] Q1 (Main board piece):

<Question> How many pieces are on the main board?

<Analysis> Pieces currently on the board: ...

Piece 2 (orange) around position (5, 4); ...

There are three pieces on the main board

<Answer>3.

[Medium] Q4 (Removed piece rotation

feasibility) :

<Question> Can the removed piece fit back

into the main board by only rotation? What

rotation(s) work? Options: ..

<Analysis> Dimension Analysis:

Hole dimensions: 8x5; Piece dimensions: 8x5.

Based on dimensions, these rotations

(clockwise) might work: [0, 180]

Test each rotation: ...

Summary: Valid rotations found: [0]

<Answer>Option 8. rotate 0 degrees

# S神经系统

![](images/c9fa790cbe37e5911fdf9c14272fcb5a610e32e7aa8ca334a540f825b6ab5a82.jpg)

Introduction: Fill the empty cells with the four colors—red, green, blue, magenta, so that each row, column, and $2 \times 2$ subgrid contains every color exactly once.

Game Category: Multi-step Reasoning

[Easy] Q1 (Position color identification):

<Question> What color is at (3,4)? Options:

A.red, B.green, C.blue, D.magenta

<Analysis> From the image we can see the color at Position (3,4) is red.

<Answer> A. red.

[Hard] Q5 (Guided-position color deduction):

<Question> After determining colors at

positions (1,4), (2,2), what color should be at

position (2,4)? Options: A.red, B.green, C.blue, D.magenta

<Analysis> Deductive reasoning process:

Step 1: Position (1,4): ... So it must be blue.

Step 2: Position (2,2): ... So it must be green.

Step 3: Final analysis for position (2,4):...

After previous deductions, possible color

reduced to: magenta

<Answer> D. magenta.

# Sokoban

![](images/a215b87cf7d913c2be1b29b5a452a97cf8c6b6ed55e14a6e490ada54d94fb8f8.jpg)  
Figure 2 Four game examples from GameQA: 3D Reconstruction, Tangram, Sodomu, and Sokoban, each representing distinct cognitive categories. Each game displays two VQA examples consisting of: (a) current game state visualization, (b) a targeted question, and (c) step-by-step reasoning with the answer. GameQA transforms complex game-playing tasks into this structured VQA format. See Appendix J for more VQA examples of some representative games.

Introduction: You control a cartoon character who pushes brown "X" boxes onto green "X" targets. Brown tiles are walls you can't cross; light-brown tiles are open floor. Push the box onto a target to win.

Game Category: Strategic Planning

[Medium] Q3 (Player final position):

<Question> If the player moves: up, right,

down, up, up, where is the player?

<Analysis> Player position: (1,3).

Move analysis: ... Initial position: ... Move 1 -

Up: Failed ...; Final position: ...

<Answer> (1,3)

[Hard] Q6 (Minimum move count):

<Question> What's the minimum number of

moves needed to solve the puzzle? Options:

[1] 7; [2] 12; [3] 3; [4] 6; [5] 5; [6] 2; ...

<Analysis> Player position: (1,3) Box position:

(2,2) Target position: (1,1).

Step by step solution analysis:

Player (1,3) to (2,3)... Player (2,3) to (2,2) (box

(2,2) to (2,1))... Player(3,2) to (3,1)... Player

(3,1) to (2,1) (box (2,1) to (1,1)) Total

moves: 5. Answer: Option5. 5.

<Answer>Option5.5.

Prediction Task" and "Strategy Optimization Task", with the detailed descriptions shown in Appendix E.1.

For a task, we use LLMs to assist the design and refinement of the QA template, an example prompt for such refinement is illustrated in Figure 1 (Step 2). Additionally, LLMs can also design tasks, with example prompt shown in Appendix G.2. In summary, designing a QA template is equivalent to extracting one type of reasoning pattern from the video game code obtained in the first step. The third step then further instantiates these reasoning task templates into a dataset.

# 2.3 Data engine construction for data samples generation

The third step is data engine construction for samples generation. The data engine is a program that, when executed, automatically generates task instances in batch according to task templates. LLMs are used to assist in constructing the data engine's code based on the game code obtained in the first step. The prompt for guiding LLMs to construct the data engine code is shown in Figure 1 (step 3). The pseudocode for the data engine is shown in Figure 1 (step 3). Taking Sokoban's state prediction task as an example, the data engine program is constructed from four core modules:

- Game environment initialization: This module randomly generates Sokoban environment by directly reusing the initialization functions from the Step 1's game code. Sokoban environment includes the position of player, boxes, goals and walls.   
- Proposing task instance: This module proposes a state prediction task instance by generating a multi-step random movement sequence.

![](images/7e713a3eef10d66e4237781f4fcd62ccb801dadcf76ecb0ec19190766ef7460d.jpg)  
Figure 3 Overview of the GameQA dataset. The 30 games in GameQA can be classified into four categories based on the core abilities required to solve game tasks. Appendix F.2 provides definitions of the four game categories. Games chose as Out-of-Domain are not used for training; instead, they are used to test the generalization performance after the model has been trained on In-Domain games.

- Solving task instance: This module generates a solution using a corresponding algorithm. For state prediction task, the algorithm simulates each action by reusing the movement logic from game code of Step 1, which handles all situations of movement including collisions and box pushing.   
- QA data construction: This module fills the task templates. The movement sequence will be filled into the question template to become a question instance. The state transition trajectory will be filled into the answer template to become an answer instance.

# 2.4 Data quality assessment and data augmentation

We implemented manual quality verifications at each step of our method.

Game code verification: We verify the correctness of the game code generated by the models by manually running the game program. If errors are found, including vision issues in the game display or code logic errors, they are fed back to the LLM for regeneration. For complex game features that LLMs cannot generate correctly, we retrieved relevant open-source code and provided it to the LLM. Games created with external code are shown in Appendix E.4.

Data engine validation: During the data engine development process, LLMs generate an initial version based on the prompt. This version is then manually tested. If the generated reasoning questions and answers do not conform to the templates from the second step or contain errors, the LLM will be instructed to regenerate the data engine.

Table 2 GameQA train and test set statistics summary. The full table is in Appendix F.1.   

<table><tr><td>Statistic Category</td><td>Train Set</td><td>Test Set</td></tr><tr><td>Total Games</td><td>20</td><td>30</td></tr><tr><td>In-Domain</td><td>20</td><td>20</td></tr><tr><td>Out-of-Domain</td><td>0</td><td>10</td></tr><tr><td>Total Tasks</td><td>102</td><td>158</td></tr><tr><td>Total Questions</td><td>126,760</td><td>15,047</td></tr><tr><td>Multiple Choice</td><td>86,520</td><td>10,518</td></tr><tr><td>Avg. Choices</td><td>7.10</td><td>7.05</td></tr><tr><td>Fill-in-the-Blank</td><td>40,240</td><td>4,529</td></tr><tr><td>Unique Images</td><td>74,620</td><td>8,620</td></tr></table>

Data augmentation and filtering: Once the data engine is constructed, data samples can be easily generated in batch, yet the answer processes often exhibit repetitive patterns. We therefore employed LLM paraphrasing to perform data augmentation, detailed in Appendix F.3. Additionally, we applied data filtering to ensure the augmented samples were correct, of appropriate length, and without excessive textual repetition. Details of our data quality assurance process can be found in Appendix F.4. Time spent on these main steps for each game in GameQA can be found in Appendix E.3.

# 3 The GameQA dataset

The GameQA dataset is a visual-language question answering dataset that transforms game-playing tasks into a Visual Question Answering (VQA) format, as shown in Figure 2. GameQA includes 30 games, 158 distinct tasks and approximately 140K questions in total.

Dataset composition: The dataset has 30 different games classified into four categories based on the core abilities required to solve game tasks: "3D Spatial Perception and Understanding", "Pattern Recognition and Matching", "Multi-step Reasoning", and "Strategic Planning", detailed descriptions shown in Appendix F.2. The criteria on choosing these 30 games are listed in Appendix E.2. As illustrated in Figure 3, the 30 games are divided into two sets: 20 In-Domain games for model training and 10 Out-of-Domain games for testing generalization performance. The Out-of-Domain set was held out during training.

Dataset question types: To ensure the final answers are verifiable, all questions in the dataset are either multiple-choice or fill-in-the-blank. The multiple-choice questions typically have 7-8 options, while the fill-in-the-blank questions require simple answers such as numbers or coordinates, with detailed statistics shown in Table 2.

Difficulty levels: Each task is assigned with one of three difficulty levels of questions ("QA Level"). And data samples are divided into 3 difficulty levels based on image complexity ("Plot Level"), namely game state or grid size, controlled by code parameters, providing three perception and reasoning difficulty levels for the tasks. Examples demonstrating these difficulty levels are in Appendix J. The training and evaluation result about difficulty are shown in Table 12 and Table 6.

# 4 Game-RL: RL on game tasks

# 4.1 RL algorithm

We use the Group Relative Policy Optimization (GRPO) [25] as our algorithm. We use the standard formulation from DeepSeek, with the loss function shown below and the hyperparameters shown in Appendix C.3.2.

$$
\begin{array}{l} \mathcal {J} _ {\mathrm {G R P O}} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ] \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| o _ {i} |} \sum_ {t = 1} ^ {| o _ {i} |} \frac {1}{| t - o _ {i} |}. \\ \left\{\min \left[ \frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})} \hat {A} _ {i, t}, \mathrm {c l i p} \left(\frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})}, 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {i, t} \right] - \beta \mathbb {D} _ {K L} [ \pi_ {\theta} | | \pi_ {\mathrm {r e f}} ] \right\} \\ \end{array}
$$

# 4.2 Reward design

LLM as a judge to give reward. Given the high diversity of responses of our dataset, traditional rule-based judge methods (e.g. text matching) suffer from insufficient accuracy in answer matching. We chose the quantized version of Qwen2.5-32B-AWQ [23] as an evaluator model to assist in scoring the generated outputs.

Outcome reward design of RL The reward signal used for training was based solely on the correctness of the model's final answer. Both the reference answer and the answer generated by the policy model are inputted into this evaluator model. The evaluator model then determines whether the generated answer matches the reference answer. The reward allocation adheres to a strict binary principle: a reward of 1 is assigned if they match, and a reward of 0 is assigned otherwise.

# 5 Experiment

# 5.1 Experiment settings

The detailed description of data preparation, training processes, and evaluation method can be found in Appendix C.

Hyperparameters of GRPO training We rollout 12 samples per questions. The model was trained for one epoch. The learning rate was 2e-7, with a $5\%$ warm-up. The clipping value $\epsilon$ was set to 0.2 and the KL-divergence coefficient $\beta$ was set to 0.04.

Table 3 Evaluation results on general vision benchmarks. These results show that game-only training enhances general reasoning capability of VLMs. We fine-tuned four VLMs (InternVL2.5-8B [3], InternVL3-8B [4], Qwen2.5-VL-7B [2], and LLaVA-OneVision-7B [12]) on 5K GameQA samples using GRPO, resulting in the models Game-RL-InternVL2.5-8B, Game-RL-InternVL3-8B, Game-RL-Qwen2.5-VL-7B, and Game-RL-LLaVA-OV-7B. The percentage of performance improvements compared to the vanilla model is denoted by $(\uparrow)$ . Best performance per section is indicated in bold.   

<table><tr><td>Models</td><td>Avg. (†)</td><td>MathVista</td><td>MathVerse</td><td>MMBench</td><td>MMMU</td><td>CharXiv</td><td>MathVision</td><td>MMMU-Pro</td></tr><tr><td colspan="9">Baseline</td></tr><tr><td>Random</td><td>14.84</td><td>17.90</td><td>12.74</td><td>26.37</td><td>24.67</td><td>0.00</td><td>10.03</td><td>12.19</td></tr><tr><td colspan="9">Proprietary Multimodal Large Language Models</td></tr><tr><td>GPT-4o</td><td>56.9</td><td>63.8</td><td>50.2</td><td>86.0</td><td>69.1</td><td>47.1</td><td>30.4</td><td>51.9</td></tr><tr><td>Claude-3.5-Sonnet</td><td>59.5</td><td>67.7</td><td>56.8</td><td>78.5</td><td>68.3</td><td>60.2</td><td>33.3</td><td>51.5</td></tr><tr><td>Gemini-2.5-Pro</td><td>73.1</td><td>77.7</td><td>65.9</td><td>88.3</td><td>79.7</td><td>62.9</td><td>66.0</td><td>71.2</td></tr><tr><td colspan="9">Open-Source Multimodal Large Language Models</td></tr><tr><td>Qwen2.5-VL-32B</td><td>60.70</td><td>77.40</td><td>60.41</td><td>88.13</td><td>63.83</td><td>47.50</td><td>33.80</td><td>53.83</td></tr><tr><td>Ovis2-34B</td><td>57.91</td><td>71.50</td><td>53.71</td><td>88.73</td><td>60.91</td><td>49.10</td><td>35.93</td><td>45.48</td></tr><tr><td>InternVL2.5-38B</td><td>55.62</td><td>68.60</td><td>48.38</td><td>86.93</td><td>57.53</td><td>41.50</td><td>39.40</td><td>46.98</td></tr><tr><td>LLaVA-OV-72B</td><td>49.51</td><td>58.60</td><td>46.85</td><td>83.13</td><td>52.74</td><td>35.20</td><td>33.87</td><td>36.18</td></tr><tr><td>Qwen2.5-VL-72B</td><td>60.95</td><td>75.50</td><td>56.87</td><td>86.80</td><td>65.34</td><td>48.10</td><td>40.60</td><td>53.45</td></tr><tr><td>InternVL2.5-78B</td><td>57.96</td><td>70.20</td><td>52.34</td><td>88.33</td><td>61.84</td><td>42.70</td><td>39.93</td><td>50.38</td></tr><tr><td>InternVL2.5-8B</td><td>45.89</td><td>57.50</td><td>36.04</td><td>81.93</td><td>47.96</td><td>31.70</td><td>28.87</td><td>37.25</td></tr><tr><td>Game-RL-InternVL2.5-8B</td><td>47.91 (+2.02)</td><td>61.70</td><td>37.11</td><td>83.87</td><td>50.06</td><td>32.00</td><td>31.93</td><td>38.69</td></tr><tr><td>InternVL3-8B</td><td>54.48</td><td>69.10</td><td>50.10</td><td>86.00</td><td>57.88</td><td>39.10</td><td>35.33</td><td>43.84</td></tr><tr><td>Game-RL-InternVL3-8B</td><td>55.88 (+1.40)</td><td>73.00</td><td>50.71</td><td>86.20</td><td>58.34</td><td>39.90</td><td>37.93</td><td>45.10</td></tr><tr><td>Qwen2.5-VL-7B</td><td>49.94</td><td>66.80</td><td>45.08</td><td>83.67</td><td>49.01</td><td>37.70</td><td>30.80</td><td>36.49</td></tr><tr><td>Game-RL-Qwen2.5-VL-7B</td><td>52.27 (+2.33)</td><td>68.20</td><td>47.97</td><td>83.53</td><td>50.53</td><td>42.70</td><td>33.07</td><td>39.89</td></tr><tr><td>LLaVA-OV-7B</td><td>41.23</td><td>55.60</td><td>33.05</td><td>81.13</td><td>41.07</td><td>27.10</td><td>23.40</td><td>27.26</td></tr><tr><td>Game-RL-LLaVA-OV-7B</td><td>42.27 (+1.04)</td><td>58.20</td><td>34.92</td><td>82.53</td><td>41.31</td><td>27.30</td><td>23.07</td><td>28.58</td></tr></table>

Table 4 GameQA's generalization is competitive compared to outstanding geometry visual reasoning datasets. Based on Qwen2.5-VL-7B, we applied the same GRPO method on 5k GameQA samples, 8k samples from MAVIS, 8k Multimodal-Open-R1 samples, 8k samples from MultiMath respectively, a mixture of 5k GameQA and 8k MultiMath samples, to conduct comparative training.   

<table><tr><td rowspan="2">Models</td><td rowspan="2">Avg. (↑)</td><td colspan="4">Out of Domain Games</td><td rowspan="2">Avg. (↑)</td><td colspan="7">General Vision Benchmarks</td></tr><tr><td>3D Spatial Perc. &amp; Under.</td><td>Pattern Recog. &amp; Matching</td><td>Multi-step Reasoning</td><td>Strategic Planning</td><td>Math Vista</td><td>Math Verse</td><td>MMBench</td><td>MMMU</td><td>CharXiv</td><td>Math Vision</td><td>MMMU-Pro</td></tr><tr><td>Qwen2.5-VL-7B</td><td>27.09</td><td>23.60</td><td>29.20</td><td>26.21</td><td>29.34</td><td>49.94</td><td>66.80</td><td>45.08</td><td>83.67</td><td>49.01</td><td>37.70</td><td>30.80</td><td>36.49</td></tr><tr><td>+ MAVIS (GRPO)</td><td>27.61 (+0.52)</td><td>26.80</td><td>28.25</td><td>28.42</td><td>26.98</td><td>51.53 (+1.59)</td><td>67.90</td><td>46.16</td><td>83.62</td><td>50.45</td><td>39.20</td><td>34.98</td><td>38.42</td></tr><tr><td>+ Multimodal-Open-R1 (GRPO)</td><td>28.33 (+1.24)</td><td>24.87</td><td>27.86</td><td>29.93</td><td>30.64</td><td>51.86 (+1.92)</td><td>67.63</td><td>48.09</td><td>83.78</td><td>49.78</td><td>40.20</td><td>34.89</td><td>38.65</td></tr><tr><td>+ MultiMath (GRPO)</td><td>28.38 (+1.29)</td><td>28.10</td><td>28.45</td><td>27.45</td><td>29.53</td><td>52.81 (+2.87)</td><td>69.36</td><td>47.99</td><td>84.13</td><td>53.44</td><td>40.83</td><td>33.92</td><td>39.91</td></tr><tr><td>+ GameQA (GRPO)</td><td>29.87 (+2.78)</td><td>27.00</td><td>28.52</td><td>31.49</td><td>32.46</td><td>52.31 (+2.37)</td><td>68.70</td><td>48.72</td><td>83.16</td><td>50.21</td><td>41.40</td><td>34.27</td><td>39.74</td></tr><tr><td>+ GameQA, MultiMath (GRPO)</td><td>30.93 (+3.84)</td><td>28.20</td><td>28.28</td><td>34.26</td><td>32.99</td><td>53.23 (+3.29)</td><td>69.20</td><td>48.02</td><td>84.73</td><td>53.21</td><td>42.10</td><td>34.47</td><td>40.89</td></tr></table>

Benchmarks We evaluated the models on a set of vison-language reasoning benchmarks consisting of our GameQA test set and public general benchmarks.

GameQA benchmark This test set includes around 500 question-answer pairs for each of 30 games, totaling 15,047 samples.   
- General benchmarks We used the MMMU validation set [30] for testing general multimodal understanding, and included MMMU-Pro [31], which features 10-option multiple-choice questions. To assess mathematical

![](images/dc2ede1622d2370983c456e2e350d3ce9336ec7b356afde622be3b68da3a7f82.jpg)  
Figure 4 The scaling effect of training data quantity on general vision benchmarks for Qwen2.5-VL-7B-Instruct. The model was trained on a total of 20k samples (20 games) and evaluated every 1,000 samples. To clearly demonstrate the upward trend, the results are divided into three stages and presented using bin averaging (as described in Section D.3).

reasoning in visual contexts, we used MathVista [19] (testmini, 1,000 samples), MathVerse [34] (testmini, 3,940 samples), and MathVision [28] (open subset, 3,040 questions). For general visual understanding, we adopted MMBench [15] (validation set), and for chart-based reasoning, we used CharXiv [29].

# 5.2 Main results

By analyzing the models' performance on in-domain tasks, related out-of-domain game tasks, and general vision-language benchmarks, we can find that:

Game-only training enhances general reasoning capability of VLMs Training on the GameQA dataset significantly improved performance on the GameQA test set. Moreover, models trained on GameQA exhibited strong generalization to unseen games, with accuracy improvements ranging from $1.16\%$ to $3.82\%$ (Table 5). On broader general vision benchmarks, these models showed robust generalization, achieving consistent performance gains across all general vision reasoning tasks. These results suggest that the models successfully learned transferable visual understanding and reasoning abilities from the GameQA dataset (see Table 3).

GameQA's out-of-domain generalization is competitive compared to outstanding geometry visual reasoning datasets. To better understand the GameQA dataset's advantages, we compared it with the outstanding geometry visual reasoning datasets, include MAVIS [35], Multimodal-Open-R1 [16] and MultiMath [22]. MAVIS includes various geometry and function problems. Multimodal-Open-R1 is a geometry-centered dataset. MultiMath is a comprehensive and diverse multimodal mathematical dataset. The results (Table 4) show that despite having fewer training samples (5k vs. 8k) that are also out-of-domain for geometry and function tasks, the GameQA-trained model is competitive compared to its counterparts trained on geometry or function data, where general vision benchmarks would be considered in-domain. These results suggest that GameQA enables stronger out-of-domain generalization, even when using less data from a mismatched domain.

GameQA enhances model performance in mixed training. We trained Qwen2.5-VL-7B using GRPO on a mixture of 5k GameQA and 8k MultiMath samples, as shown in Table 4. This results suggest that GameQA can bring extra benefits when mix training with other dataset.

![](images/056bf837701b4984252302bd6127a34f80cb63e5f871fe1f552b3fcf04c97fcc.jpg)  
Figure 5 The scaling effect of GameQA. As VLMs are trained on an increasing number of distinct games, their performance on general visual benchmarks improves. Game selection is shown in Table 7.

![](images/3074e5e9fa3621c5ac79730c36b3f45ec9eba81e2db52b2f3354006d37dc87b3.jpg)  
Figure 6 Impact of GRPO on visual perception performance on general visual benchmarks. Two pie charts and one example below illustrate how performance improves after GRPO.

# 5.3 Scaling effect of game samples on generalization

We trained the Qwen2.5-VL-7B model on a GameQA subset of 20,000 samples from 20 games using the GRPO method. As shown in Figure 4, the model's performance score demonstrates a overall upward trend on 7 general vision benchmarks as the amount of training data increases. This indicates that scaling up training on GameQA data effectively enhances the VLM's general reasoning abilities.

# 5.4 Scaling effect of game diversity on generalization

With around 5,000 total training samples, we trained the Qwen2.5-VL-3B on GameQA subsets with 4 and 20 distinct games (Table 7). The results (Figure 5) show a positive correlation between game diversity and generalization ability. This suggests that scaling game diversity makes better generalization, enabling the model to acquire more robust visual understanding and reasoning abilities.

# 5.5 Qualitative analysis

To confirm that GRPO substantially enhances visual perception and text reasoning abilities of models, we manually analyzed 790 cases randomly sampled from the results of InternVL2.5-8B, containing responses before and after GRPO. The results (Figure 6) confirm that after GRPO, the model demonstrates improved visual recognition of image elements and performs more precise reasoning. More statistics and cases are in Appendix B.2.

In addition, our qualitative analysis of model performances across four game categories reveals common behaviors and challenges, detailed in Appendix I.

# 5.6 Advanced vision-language models perform notably worse than humans on the GameQA benchmark

Both leading open-source and proprietary models achieve average accuracy levels considerably lower than those of human (Table 5). This clear difference highlights the difficulty and high requirements of the GameQA benchmark, requiring not only accurate visual comprehension of game scenes but also the ability to carry out multi-step logical reasoning. The training and evaluation result about difficulty are shown in Table 12 and Table 6. Furthermore, our qualitative analysis and case study in Appendix I reveal that even the most

Table 5 Evaluation results on GameQA benchmark. The experimental setup is consistent with Table 3: four VLMs were fine-tuned on 5K GameQA samples using GRPO. We evaluate the models on both in-domain and out-of-domain game categories. The percentage improvement over the vanilla model is marked with a green upward arrow (↑). The best performance for each category is presented in bold.   

<table><tr><td rowspan="2">Models</td><td rowspan="2">Avg. (↑)</td><td colspan="4">In Domain Games</td><td rowspan="2">Avg. (↑)</td><td colspan="4">Out of Domain Games</td></tr><tr><td>3D Spatial Perc. &amp; Under.</td><td>Pattern Recog. &amp; Matching</td><td>Multi-step Reasoning</td><td>Strategic Planning</td><td>3D Spatial Perc. &amp; Under.</td><td>Pattern Recog. &amp; Matching</td><td>Multi-step Reasoning</td><td>Strategic Planning</td></tr><tr><td colspan="11">Baselines</td></tr><tr><td>Human</td><td>84.75</td><td>85.18</td><td>80.74</td><td>84.46</td><td>88.62</td><td>81.61</td><td>79.17</td><td>73.81</td><td>81.27</td><td>92.19</td></tr><tr><td>Random</td><td>11.90</td><td>11.69</td><td>12.04</td><td>10.24</td><td>13.61</td><td>9.68</td><td>7.11</td><td>9.50</td><td>11.83</td><td>10.27</td></tr><tr><td colspan="11">Proprietary Multimodal Large Language Models</td></tr><tr><td>GPT-4o</td><td>40.52</td><td>32.01</td><td>34.81</td><td>50.67</td><td>44.59</td><td>43.81</td><td>48.90</td><td>36.91</td><td>48.58</td><td>40.87</td></tr><tr><td>Claude-3.5-Sonnet</td><td>47.69</td><td>37.41</td><td>43.16</td><td>56.09</td><td>54.11</td><td>50.34</td><td>51.30</td><td>43.62</td><td>60.42</td><td>46.01</td></tr><tr><td>Claude-4-Sonnet</td><td>46.58</td><td>31.12</td><td>39.73</td><td>66.90</td><td>48.57</td><td>55.16</td><td>45.60</td><td>56.58</td><td>63.28</td><td>55.17</td></tr><tr><td>Gemini-2.5-Pro</td><td>58.95</td><td>46.93</td><td>52.79</td><td>74.62</td><td>61.46</td><td>67.60</td><td>57.60</td><td>77.37</td><td>77.62</td><td>57.80</td></tr><tr><td colspan="11">Open-Source Multimodal Large Language Models</td></tr><tr><td>Ovis2-8B</td><td>24.98</td><td>19.92</td><td>24.43</td><td>27.21</td><td>28.37</td><td>26.99</td><td>29.70</td><td>20.70</td><td>34.14</td><td>23.41</td></tr><tr><td>InternVL3-9B</td><td>26.89</td><td>21.86</td><td>22.53</td><td>32.54</td><td>30.65</td><td>26.73</td><td>25.20</td><td>30.38</td><td>32.14</td><td>19.18</td></tr><tr><td>LLaMA3.2-11B-Vision</td><td>19.69</td><td>19.12</td><td>16.48</td><td>21.30</td><td>21.86</td><td>18.04</td><td>18.40</td><td>14.92</td><td>17.73</td><td>21.11</td></tr><tr><td>Qwen2.5-VL-32B</td><td>34.09</td><td>28.26</td><td>30.99</td><td>40.27</td><td>36.83</td><td>35.97</td><td>32.90</td><td>33.02</td><td>44.03</td><td>33.94</td></tr><tr><td>Ovis2-34B</td><td>34.53</td><td>27.92</td><td>32.72</td><td>39.46</td><td>38.03</td><td>35.29</td><td>35.50</td><td>34.20</td><td>38.71</td><td>32.75</td></tr><tr><td>InternVL2.5-38B</td><td>30.04</td><td>23.39</td><td>25.86</td><td>36.82</td><td>34.08</td><td>32.42</td><td>30.60</td><td>33.96</td><td>39.35</td><td>25.79</td></tr><tr><td>InternVL3-38B</td><td>35.23</td><td>28.33</td><td>31.76</td><td>41.89</td><td>38.96</td><td>38.69</td><td>33.30</td><td>43.62</td><td>50.09</td><td>27.75</td></tr><tr><td>LLaVA-OV-72B</td><td>24.87</td><td>19.92</td><td>24.88</td><td>27.72</td><td>26.95</td><td>28.32</td><td>26.80</td><td>23.52</td><td>32.87</td><td>30.11</td></tr><tr><td>Qwen2.5-VL-72B</td><td>37.63</td><td>29.47</td><td>32.85</td><td>45.76</td><td>42.42</td><td>39.22</td><td>35.90</td><td>37.38</td><td>46.86</td><td>36.75</td></tr><tr><td>InternVL2.5-78B</td><td>32.35</td><td>27.15</td><td>28.84</td><td>39.53</td><td>33.90</td><td>35.30</td><td>32.80</td><td>37.26</td><td>42.41</td><td>28.75</td></tr><tr><td>InternVL3-78B</td><td>38.00</td><td>33.15</td><td>33.03</td><td>46.60</td><td>39.20</td><td>39.74</td><td>34.90</td><td>40.70</td><td>50.95</td><td>32.43</td></tr><tr><td>InternVL2.5-8B</td><td>22.22</td><td>20.39</td><td>17.18</td><td>25.34</td><td>25.97</td><td>20.05</td><td>20.80</td><td>22.45</td><td>18.88</td><td>18.07</td></tr><tr><td>Game-RL-InternVL2.5-8B</td><td>29.44 (+7.22)</td><td>26.74</td><td>26.05</td><td>29.51</td><td>35.44</td><td>23.87 (+3.82)</td><td>25.00</td><td>25.12</td><td>24.91</td><td>20.45</td></tr><tr><td>InternVL3-8B</td><td>26.51</td><td>22.53</td><td>21.91</td><td>30.18</td><td>31.43</td><td>27.64</td><td>29.60</td><td>27.44</td><td>29.62</td><td>23.91</td></tr><tr><td>Game-RL-InternVL3-8B</td><td>33.09 (+6.58)</td><td>27.94</td><td>28.52</td><td>36.81</td><td>39.07</td><td>28.80 (+1.16)</td><td>29.20</td><td>25.31</td><td>34.59</td><td>26.09</td></tr><tr><td>Qwen2.5-VL-7B</td><td>25.78</td><td>22.58</td><td>21.92</td><td>25.21</td><td>33.40</td><td>27.09</td><td>23.60</td><td>29.20</td><td>26.21</td><td>29.34</td></tr><tr><td>Game-RL-Qwen2.5-VL-7B</td><td>32.12 (+6.34)</td><td>26.80</td><td>26.88</td><td>33.34</td><td>41.45</td><td>30.51 (+3.42)</td><td>27.10</td><td>31.56</td><td>31.24</td><td>32.13</td></tr><tr><td>LLaVA-OV-7B</td><td>21.79</td><td>18.84</td><td>19.69</td><td>21.24</td><td>27.38</td><td>20.39</td><td>21.00</td><td>20.19</td><td>20.13</td><td>20.27</td></tr><tr><td>Game-RL-LLaVA-OV-7B</td><td>33.49 (+11.70)</td><td>29.87</td><td>31.10</td><td>30.96</td><td>42.03</td><td>23.34 (+2.95)</td><td>27.20</td><td>20.05</td><td>23.55</td><td>22.57</td></tr></table>

advanced models currently struggle to match human-level understanding, particularly in tasks demanding deep reasoning. More experiments can be found in Appendix D.

# 6 Related work

Multimodal reasoning data construction Currently, the data construction methods are mainly divided into two categories: human expert supervision and automated synthesis. Peng et al. [22] and Lu et al. [17] collect visual reasoning problems from textbooks, Lu et al. [18] constructs datasets through labeling by STEM students, but they are limited by the scarcity of high-quality data sources and the high cost of manual annotation. Gao et al. [6], He et al. [8], Lu et al. [17], Shi et al. [26] uses expert models to generate reasoning processes, but the results are limited by the performance of the expert model. Trinh et al. [27] and Zhang et al. [35] synthesize geometric reasoning data through procedural methods, but these methods are often designed for specific domains and have high transfer costs. Table 1 provides a comparison of existing vision language reasoning datasets.

Using game data to enhance VLMs reasoning capabilities Game environments provide well-defined rules and mechanics that are easy to verify. However, existing work has not fully leveraged the potential of game environments in visual reasoning data construction. Reed et al. [24] trains a general agent by tokenizing game images and action sequences, but it is difficult to generalize on out-of-domain game tasks and this method relies on expensive expert trajectory data; Li et al. [13], Paglieri et al. [21], Zhang et al. [32, 33] all established gaming environments for Vision-Language Models, but these were used exclusively for evaluation purposes. These limitations indicate that how to effectively use game data to enhance the reasoning ability of visual language models remains a critical problem that needs to be addressed. Table 1 provides a comparison of existing game reasoning benchmarks.

# 7 Conclusion

To explore broader training scenarios and resources for vision-language RL, we propose Game-RL to construct game tasks for VLMs' RL training. We also propose the novel Code2Logic approach adapting game code to synthesize diverse game reasoning task data, thus obtaining the GameQA dataset for Game-RL. Multiple VLMs trained through RL solely on GameQA achieved performance improvements across diverse general vision-language reasoning benchmarks. This not only demonstrates the value of Game-RL for enhancing VLMs' general reasoning abilities, but also suggests that video games may serve as valuable scenarios and resources to boost VLMs' general reasoning.

# References

[1] Anthropic. Claude 3.5 sonnet model card addendum, 2024. URL https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf.   
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.   
[3] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.   
[4] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Internv13: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.   
[5] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training, 2025. URL https://arxiv.org/abs/2501.17161.   
[6] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.   
[7] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[8] Wei He, Zhiheng Xi, Wanxu Zhao, Xiaoran Fan, Yiwen Ding, Zifei Shan, Tao Gui, Qi Zhang, and Xuanjing Huang. Distill visual chart reasoning ability from llms to mllms. arXiv preprint arXiv:2410.18798, 2024.   
[9] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuzhhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/abs/2106.09685.   
[10] Jin Jiang, Yuchen Yan, Yang Liu, Yonggang Jin, Shuai Peng, Mengdi Zhang, Xunliang Cai, Yixin Cao, Liangcai Gao, and Zhi Tang. Logicpro: Improving complex logical reasoning via program-guided learning. arXiv preprint arXiv:2409.12929, 2024.   
[11] Zhiqiang Jiang, Yihuai Lan, Minggao Zhang, Haozhe Feng, Wenyu Liu, Junwei Dong, Ze Ji, Xin Zhao, and Xing Xu. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models, 2024.   
[12] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer, 2024. URL https://arxiv.org/abs/2408.03326.   
[13] Chenglin Li, Qianglong Chen, Zhi Li, Feng Tao, and Yin Zhang. Vcbench: A controllable benchmark for symbolic and abstract challenges in video cognition. arXiv preprint arXiv:2411.09105, 2024.   
[14] Junteng Liu, Yuanxiang Fan, Zhuo Jiang, Han Ding, Yongyi Hu, Chi Zhang, Yiqi Shi, Shitong Weng, Aili Chen, Shiqi Chen, et al. Synlogic: Synthesizing verifiable reasoning data at scale for learning logical reasoning and beyond. arXiv preprint arXiv:2505.19641, 2025.

[15] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.   
[16] lmms lab. Multimodal-open-r1-8k-verified. https://huggingface.co/datasets/lmmss-lab/multimodal-open-r1-8k-verified, 2025.   
[17] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.   
[18] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.   
[19] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024.   
[20] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.   
[21] Davide Paglieri, Bartlomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kuciński, Lerrel Pinto, Rob Fergus, et al. Balrog: Benchmarking agentic llm and vlm reasoning on games. arXiv preprint arXiv:2411.13543, 2024.   
[22] Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024.   
[23] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.   
[24] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.   
[25] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[26] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.   
[27] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476-482, 2024.   
[28] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804, 2024.   
[29] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. arXiv preprint arXiv:2406.18521, 2024.   
[30] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Chen Cui, Tanmay Rao, Human-Machine interfacing, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28039-28064, 2024.   
[31] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Meiqi Yin, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.

[32] Alex L Zhang, Thomas L Griffiths, Karthik R Narasimhan, and Ofir Press. Videogamebench: Can vision-language models complete popular video games? arXiv preprint arXiv:2505.18134, 2025.   
[33] Haoran Zhang, Hangyu Guo, Shuyue Guo, Meng Cao, Wenhao Huang, Jiaheng Liu, and Ge Zhang. Ing-vp: Mllms cannot play easy vision-based games yet. arXiv preprint arXiv:2410.06555, 2024.   
[34] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.   
[35] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv e-prints, pages arXiv-2407, 2024.   
[36] Jun Zhao, Jingqi Tong, Yurong Mou, Ming Zhang, Qi Zhang, and Xuanjing Huang. Exploring the compositional deficiency of large language models in mathematical reasoning through trap problems. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16361-16376, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.915. URL https://aclanthology.org/2024.emnlp-main.915/.   
[37] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https://arxiv.org/abs/2306.05685.

# Appendix

# Appendix Contents

A Limitations and future work 16   
B More analysis 16

B.1 GameQA evaluation results by difficulty 16   
B.2 Error types analysis 16

C Experiment details 17

C.1 Human and random baselines 17   
C.2 Data preparation 18

C.2.1 SFT data preparation 18   
C.2.2 Reinforcement Learning Data preparation 18

C.3 Training details 19

C.3.1 Models 20   
C.3.2 Training hyperparameters 20   
C.3.3 Compute resources 20

C.4 Inference and evaluation 21

D More Experiments 21

D.1 SFT experiments 21   
D.2 Training data difficulty experiment 23

D.3 Data scale experiment 23

E Approach details 23

E.1 Task category 24   
E.2 Selection criteria of 30 games in GameQA 24   
E.3 Time spent on the main steps of Code2Logic for each game 24   
E.4 Games generated using external open-source code 25

F More information about the GameQA Dataset 25

F.1 The GameQA dataset statistics 25   
F.2 Definition of each category 25   
F.3 Data augmentation 26   
F.4 Data quality assurance 26

G Details on Sokoban task synthesis 26

G.1 Sokoban QA templates 26   
G.2 Supplementary prompt 27

H Prompts 28

H.1 Data augmentation 29   
H.2 Training 29   
H.3 Inference 30   
H.4 Evaluation 30

1 Case study on model performance on GameQA 30   
J Details on the 30 games and more example data samples in the GameQA dataset 34

J.1 3D Spatial Perception and Understanding 36

J.1.1 3D Reconstruction 36   
J.1.2 3D Maze 39   
J.1.3 Rubik's Cube 42   
J.1.4 Pyramid Chess 42   
J.1.5 Minecraft 43

J.2 Pattern Recognition and Matching 44

J.2.1 Color Hue 44   
J.2.2 Tangram 45   
J.2.3 Freecell 48   
J.2.4 Tetris 48

J.2.5 Zuma 48  
J.2.6 Spider Solitaire 49   
J.2.7 Jewel2 49   
J.2.8 Klondike 50

# J.3 Multi-step Reasoning 51

J.3.1 Star Battle 51   
J.3.2Sudoku 53   
J.3.3 Langton's Ant 55   
J.3.4 Word Search 57   
J.3.5 2D Turing Machine 58   
J.3.6 Tents 58   
J.3.7 Rhythm Game 59   
J.3.8 Lifegame 59   
J.3.9 Minesweeper 60

# J.4 Strategic Planning 61

J.4.1 Sokoban 61   
J.4.2 Maze 63   
J.4.3 TicTacToe 65   
J.4.4 Ultra TicTacToe 66   
J.4.5 Space Invaders 67   
J.4.6 Snake 67   
J.4.7 ChessRanger 68   
J.4.8 Pacman 68

# A Limitations and future work

Using reasoning processes to conduct Supervised Fine-Tuning (SFT) has not achieved satisfactory out-of-domain generalization. Therefore, future work could explore methods to better leverage these reasoning processes to enhance the model's general capabilities, such as employing reinforcement learning based on process supervision. In addition, GameQA currently involves single-turn game question answering. Future work could focus on developing training and evaluation methods for multi-turn interactions in gaming scenarios.

# B More analysis

# B.1 GameQA evaluation results by difficulty

Table 6 Evaluation results on GameQA benchmark by difficulty. Scores are broken down by question difficulty (QA Level) and image complexity (Plot Level) within in-domain and out-of-domain game sets. The percentage of performance improvements compared to the vanilla model is denoted by $(\uparrow)$ . Best performance per section is indicated in bold.   

<table><tr><td rowspan="3">Models</td><td rowspan="3">Avg.(↑)</td><td colspan="6">In Domain Games by Difficulty</td><td rowspan="3">Avg.(↑)</td><td colspan="6">Out of Domain Games by Difficulty</td></tr><tr><td colspan="3">QA Level</td><td colspan="3">Plot Level</td><td colspan="3">QA Level</td><td colspan="3">Plot Level</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Easy</td><td>Medium</td><td>Hard</td></tr><tr><td colspan="15">Proprietary Multimodal Large Language Models</td></tr><tr><td>GPT-4o</td><td>41.56</td><td>48.10</td><td>40.63</td><td>32.95</td><td>47.69</td><td>38.60</td><td>37.80</td><td>44.03</td><td>55.63</td><td>44.45</td><td>32.05</td><td>54.49</td><td>42.98</td><td>33.71</td></tr><tr><td>Claude-3.5-Sonnet</td><td>48.90</td><td>59.26</td><td>46.07</td><td>36.70</td><td>55.49</td><td>45.97</td><td>43.80</td><td>50.94</td><td>62.80</td><td>46.15</td><td>43.80</td><td>60.58</td><td>49.19</td><td>42.26</td></tr><tr><td>Claude-4-Sonnet</td><td>48.80</td><td>58.36</td><td>50.14</td><td>42.23</td><td>56.45</td><td>48.99</td><td>47.81</td><td>56.03</td><td>66.71</td><td>55.91</td><td>45.45</td><td>62.80</td><td>54.31</td><td>50.37</td></tr><tr><td>Gemini-2.5-Pro</td><td>60.62</td><td>67.43</td><td>62.36</td><td>56.21</td><td>67.48</td><td>62.42</td><td>57.72</td><td>67.68</td><td>74.47</td><td>67.56</td><td>61.10</td><td>74.23</td><td>66.43</td><td>61.90</td></tr><tr><td colspan="15">Open-Source Multimodal Large Language Models</td></tr><tr><td>Ovis2-8B</td><td>25.56</td><td>33.67</td><td>22.80</td><td>20.21</td><td>28.30</td><td>26.12</td><td>23.69</td><td>27.37</td><td>32.41</td><td>28.87</td><td>20.90</td><td>33.96</td><td>25.56</td><td>22.06</td></tr><tr><td>InternVL3-9B</td><td>27.45</td><td>34.92</td><td>24.20</td><td>21.15</td><td>30.42</td><td>26.94</td><td>24.43</td><td>26.54</td><td>30.45</td><td>23.23</td><td>25.86</td><td>34.41</td><td>23.93</td><td>20.63</td></tr><tr><td>LLaMA3.2-11B-Vision</td><td>19.67</td><td>24.51</td><td>18.54</td><td>15.96</td><td>20.71</td><td>20.22</td><td>19.16</td><td>18.35</td><td>19.37</td><td>20.32</td><td>15.41</td><td>20.99</td><td>18.14</td><td>15.68</td></tr><tr><td>Qwen2.5-VL-32B</td><td>34.80</td><td>41.52</td><td>33.43</td><td>27.18</td><td>38.07</td><td>34.93</td><td>31.09</td><td>36.59</td><td>47.99</td><td>33.60</td><td>28.16</td><td>46.87</td><td>32.25</td><td>29.86</td></tr><tr><td>Ovis2-34B</td><td>35.32</td><td>46.36</td><td>31.81</td><td>25.11</td><td>39.57</td><td>34.70</td><td>31.70</td><td>35.40</td><td>46.86</td><td>30.02</td><td>29.22</td><td>42.95</td><td>33.76</td><td>28.87</td></tr><tr><td>InternVL2.5-38B</td><td>30.83</td><td>40.08</td><td>28.21</td><td>22.68</td><td>35.55</td><td>30.36</td><td>27.23</td><td>32.50</td><td>38.80</td><td>29.71</td><td>28.93</td><td>41.52</td><td>30.74</td><td>24.47</td></tr><tr><td>InternVL3-38B</td><td>36.07</td><td>44.05</td><td>31.73</td><td>30.60</td><td>40.70</td><td>36.47</td><td>30.20</td><td>38.78</td><td>50.00</td><td>36.14</td><td>30.17</td><td>49.03</td><td>35.62</td><td>30.86</td></tr><tr><td>LLaVA-OV-72B</td><td>25.49</td><td>35.50</td><td>21.21</td><td>17.57</td><td>27.81</td><td>24.66</td><td>23.87</td><td>28.98</td><td>38.80</td><td>23.83</td><td>24.20</td><td>34.07</td><td>27.00</td><td>25.46</td></tr><tr><td>Qwen2.5-VL-72B</td><td>38.58</td><td>47.61</td><td>36.70</td><td>27.38</td><td>43.50</td><td>37.67</td><td>33.43</td><td>39.77</td><td>52.90</td><td>37.90</td><td>28.51</td><td>49.15</td><td>37.13</td><td>32.28</td></tr><tr><td>InternVL2.5-78B</td><td>33.05</td><td>41.90</td><td>30.58</td><td>24.00</td><td>36.58</td><td>32.30</td><td>30.00</td><td>35.40</td><td>43.96</td><td>34.57</td><td>27.69</td><td>45.22</td><td>32.97</td><td>27.20</td></tr><tr><td>InternVL3-78B</td><td>38.64</td><td>47.64</td><td>35.60</td><td>30.47</td><td>43.91</td><td>37.34</td><td>34.57</td><td>40.17</td><td>52.37</td><td>35.84</td><td>32.23</td><td>50.34</td><td>39.36</td><td>29.93</td></tr><tr><td>InternVL2.5-8B</td><td>22.31</td><td>27.12</td><td>21.98</td><td>18.35</td><td>24.65</td><td>22.66</td><td>21.37</td><td>19.78</td><td>18.36</td><td>21.29</td><td>19.72</td><td>24.52</td><td>17.78</td><td>16.67</td></tr><tr><td>+ GameQA (SFT)</td><td>47.33 (+25.02)</td><td>56.59</td><td>45.36</td><td>39.79</td><td>53.73</td><td>45.50</td><td>44.76</td><td>26.10 (+6.32)</td><td>27.84</td><td>21.95</td><td>28.39</td><td>30.43</td><td>27.37</td><td>20.07</td></tr><tr><td>+ GameQA (GRPO)</td><td>29.52 (+7.21)</td><td>36.37</td><td>26.95</td><td>25.90</td><td>34.23</td><td>28.14</td><td>27.91</td><td>23.67 (+3.90)</td><td>24.41</td><td>22.92</td><td>23.67</td><td>28.44</td><td>22.85</td><td>19.33</td></tr><tr><td>InternVL3-8B</td><td>26.85</td><td>33.82</td><td>26.37</td><td>19.79</td><td>28.91</td><td>28.20</td><td>24.90</td><td>27.49</td><td>32.29</td><td>26.62</td><td>23.55</td><td>32.59</td><td>26.46</td><td>22.99</td></tr><tr><td>+ GameQA (SFT)</td><td>51.08 (+24.23)</td><td>63.08</td><td>47.66</td><td>43.22</td><td>58.93</td><td>48.69</td><td>48.74</td><td>27.35 (-0.14)</td><td>37.56</td><td>22.07</td><td>22.31</td><td>35.38</td><td>24.83</td><td>21.19</td></tr><tr><td>+ GameQA (GRPO)</td><td>33.53 (+6.68)</td><td>39.73</td><td>32.17</td><td>28.99</td><td>37.16</td><td>32.37</td><td>32.80</td><td>29.14 (+1.65)</td><td>34.36</td><td>29.41</td><td>23.67</td><td>34.81</td><td>27.31</td><td>24.85</td></tr><tr><td>Qwen2.5-VL-7B</td><td>26.02</td><td>36.25</td><td>23.11</td><td>17.75</td><td>28.29</td><td>25.69</td><td>25.67</td><td>27.25</td><td>31.87</td><td>25.83</td><td>24.03</td><td>33.73</td><td>25.38</td><td>22.12</td></tr><tr><td>+ GameQA (SFT)</td><td>49.23 (+23.21)</td><td>60.19</td><td>47.47</td><td>39.13</td><td>55.56</td><td>47.65</td><td>46.60</td><td>30.33 (+3.08)</td><td>42.00</td><td>25.35</td><td>23.55</td><td>36.86</td><td>29.29</td><td>24.29</td></tr><tr><td>+ GameQA (GRPO)</td><td>32.41 (+6.39)</td><td>42.51</td><td>30.25</td><td>23.96</td><td>35.05</td><td>32.40</td><td>31.79</td><td>30.77 (+3.52)</td><td>38.68</td><td>25.96</td><td>27.57</td><td>38.17</td><td>28.87</td><td>24.66</td></tr><tr><td>LLaVA-OV-7B</td><td>21.65</td><td>29.28</td><td>19.24</td><td>17.67</td><td>23.19</td><td>22.03</td><td>22.20</td><td>20.37</td><td>22.99</td><td>17.04</td><td>21.02</td><td>24.40</td><td>20.31</td><td>16.05</td></tr><tr><td>+ GameQA (SFT)</td><td>46.25 (+24.59)</td><td>54.62</td><td>46.81</td><td>36.70</td><td>51.65</td><td>44.55</td><td>45.03</td><td>23.39 (+3.02)</td><td>29.56</td><td>20.86</td><td>19.72</td><td>29.35</td><td>23.51</td><td>16.79</td></tr><tr><td>+ GameQA (GRPO)</td><td>33.61 (+11.95)</td><td>41.41</td><td>31.26</td><td>29.94</td><td>38.83</td><td>31.57</td><td>33.53</td><td>23.32 (+2.94)</td><td>30.15</td><td>18.25</td><td>21.43</td><td>28.90</td><td>23.51</td><td>17.04</td></tr></table>

The fine-grained difficulty gradation in the GameQA dataset enables a more systematic evaluation of the models. As shown in Table 6, when either QA Level or Plot Level increases, the models' accuracy scores generally show a noticeable decrease.

# B.2 Error types analysis

We manually checked the reasoning process of 650 GameQA cases randomly sampled from InternVL2.5-8B before and after GRPO to identify the error types. We categorize errors as 3 types: Visual Perception Error, Text Reasoning Error and both. The results are presented in Figure 9. Besides, in Figure 7 and Figure8 we supplement to Figure 6 with two examples of text reasoning improvement and another example of visual reasoning improvement after GRPO training.

# Text Reasoning Improved

Question: What is the shortest sequence of moves from position $(3, 3)$ to position $(2, 1)$ ? Options: ... [3] Up $\rightarrow$ Left $\rightarrow$ Left ...

Correct Answer: [3]

Response Before GRPO:

... 3. **Move Left**: From (3, 1), moving left to (3, 0) ... 4. **Move

Right**: From (3, 0), moving right to (3, 1) ... The answer is [2].

# Response After GRPO:

1. **Move Up**: From (3, 3) to (2, 3) ... to get closer to the target row.   
2. **Move Left**: From (2, 3) to (2, 2) ... brings the player closer to the target column.   
3. **Move Left**: From (2, 2) to (2, 1) ... places the player at the target position (2, 1) ... The answer is [3].

![](images/dab70d60fcab7c95eb8aaa4cbf39789d957ebe0b3888e720b230f0df70c69860.jpg)

![](images/6a6b610189adf81612f14e6f5aa578d7115fbc73292d6efc142969c909cae7d4.jpg)

![](images/3ba61c7ff7eb7f2b6b466451e414f3fea037d4eb5ab933bb1d9ce7a876665d07.jpg)

# Text Reasoning Improved

Question: Which point indicates birth rate equals death rate?

Correct Answer: E

Response Before GRPO:

... 4. **Point D**: The population size reaches a peak, indicating that the birth rate is equal to the death rate, resulting in a stable population size.

![](images/643f4bd3e3bee660497a6ae19140f7b5f664b7f3341569798d9dd22f362c13ab.jpg)  
Figure 7 Impact of GRPO on text reasoning performance, with two specific improvement examples.

Response After GRPO:

... 3. **Logistic Growth (Point D)**: Birth rates are still higher than death rates, but the growth rate is slowing down.

4. **Stable Population (Points E)**: The population size stabilizes. At this point, birth rates equal death rates, resulting in a stable population size.

# C Experiment details

# C.1 Human and random baselines

We included two baselines for comparison: human and random.

- Human Baseline: We selected approximately 20 questions from each of the 30 games, resulting in 623 questions. These were grouped into 30 sets and assigned to STEM undergraduates unfamiliar with the games. Each question was presented in a PowerPoint slide (Figure 10) using python-pptx $^{1}$ , and responses were collected via an online questionnaire. $^{2}$   
- Random Baseline: This represents the lower performance bound, calculated as the expected score from random guessing on multiple-choice questions, with fill-in-the-blank tasks contributing zero.

![](images/5ac649e3fc9d2b78cf50073fb3afff5e86d06f989835813b4b7737ed64a60324.jpg)

Question: What is the correct height relationship between the three numbered points?

Correct Answer: $3 < 1 = 2$

Response Before GRPO:

Point 1 is lower than Point 2 because the player must climb the ladder from Point 1 to reach Point 2 ... The answer is $1 < 2 = 3$ .

Response After GRPO:

Cube 3 is lower than cube 1. Cube 1 is at the same height as cube 2. Therefore, the correct height relationship is: $3 < 1 = 2$ .

![](images/c2d6bf72cc9a5ecb2624ec89edd0bc989bda73fa673ad5031ec7b6cb341866a0.jpg)

![](images/280ce3278577daf4c0f1fcfa02d0067150634b40d43e64213550e3274464ff06.jpg)  
Figure 8 Impact of GRPO on visual reasoning performance on GameQA.   
Figure 10 Two example PowerPoint slides demonstrating how the student participants, during the human baseline evaluation, took the tests and might write or draw on the slides.

# C.2 Data preparation

# C.2.1 SFT data preparation

- Game data preparation. First, we used our data engine to generate 5000 task instances from 20 in-domain games, obtaining 100000 samples in total. Then we performed data augmentation, as shown in Appendix F.3. Finally, we filtered these samples, as shown in Appendix F.4. 40000 samples were selected for SFT.   
- Other dataset preparation. We prepared the Geo-Multi dataset for comparing the data quality. Geo-Multi consists of 40,000 samples, with 20,000 randomly sampled from MultiMath300K [22] and another 20,000 from Geo170K [6].

# C.2.2 Reinforcement Learning Data preparation

- Game data preparation. We sampled 5000 samples from 20 in-domain games. This sample size is smaller than that used for SFT to balance performance and computational cost, as the GRPO training process is

# Visual Perception Error

Question: How many bishops are on the board?

Correct Answer: 1

Response:

The Bishops are located on the squares e3 and f7. Therefore, the answer is 2 bishops.

![](images/245f9d3d6645479f29d9be3f50c145b2341e3484548325a2696a5b0ca25e9b08.jpg)

![](images/91354de66981916e779561a8b0c92b5ffca680829ccc838a287a7c3145670ccc.jpg)

# Text Reasoning Error

Question: How many cells have been revealed?

Correct Answer: 12

Response:

We need to count the number of cells that have either numbers or are flagged. There are 4 cells with numbers. There are 3 cells flagged with "F". Therefore, the number of cells that have been revealed is $4 + 3 = 7$ .

![](images/2f929c98cc2b8f9fbb79ac84908aede09fd3024716465ffe8e6d73accf824773.jpg)  
Minesweeper Board   
Figure 9 InternVL2.5-8B error types before and after GRPO show that GRPO increases correct ratio and reduces visual and reasoning errors. Two cases show the two types of error in detail. Solid and dash lines connected to two cases means both are before GRPO.

relatively resource-intensive.

- Other dataset preparation. We sampled 4k examples from each of the geometry and function splits of MAVIS [35]. For multimodal-open-r1-8k-verified [16], we used all the samples.

To investigate the impact of game diversity on model generalization, we constructed subsets of 4, 10, and the full 20 in-domain training games. The selection process aimed to maintain representation across our four defined cognitive categories by randomly selecting games from each category to form the smaller subsets. Table 7 details the specific games included in each experimental set for this scaling analysis.

# C.3 Training details

Table 7 Selection of games for the game diversity scaling experiment across four cognitive categories.   

<table><tr><td>Game Set</td><td>3D Spatial Perception and Understanding</td><td>Pattern Recognition and Matching</td><td>Multi-step Reasoning</td><td>Strategic Planning</td></tr><tr><td>4 Games</td><td>3D Reconstruction</td><td>Tangram</td><td>Word-Search</td><td>TicTacToe</td></tr><tr><td>10 Games</td><td>3D Maze</td><td>Spider Solitaire</td><td>Tents, 2D Turing Machine</td><td>Sokoban, Space Invaders</td></tr><tr><td>20 Games</td><td>Rubik&#x27;s Cube</td><td>Freecell, Tetris, Zuma, Color Hue</td><td>Langton&#x27;s Ant, Rhythm Game, Star Battle</td><td>Maze, Ultra TicTacToe</td></tr></table>

# C.3.1 Models

We trained four VLMs, InternVL2.5-8B [3], InternVL3-8B [4], Qwen2.5-VL-7B [2], and LLaVA-OneVision-7B [12] on our data.

# C.3.2 Training hyperparameters

LoRA-based supervised fine-tuning hyperparameters In this setup, Low-Rank Adaptation (LoRA) [9] was applied to all linear layers of the language model. We trained the model for one epoch. The rank was set to 16, with alpha set to 32. The learning rate was 5e-5, including a $3\%$ warm-up period followed by a constant rate.

GRPO-based reinforcement learning In this setup, we conducted full-parameter fine-tuning of the language model while freezing the visual encoder and projection layers. We rollout 12 samples per questions. We trained the model for one epoch. The learning rate was 2e-7, with a $5\%$ warm-up. The clipping value $\epsilon$ was set to 0.2 and the KL-divergence coefficient $\beta$ was set to 0.04. More hyperparameters are listed in Table 8.

Table 8 GRPO Hyperparameters.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td>2e-7</td></tr><tr><td>Batch Size</td><td>3</td></tr><tr><td>KL-divergence Coefficient (β)</td><td>0.04</td></tr><tr><td>Number of Generations</td><td>12</td></tr><tr><td>Temperature</td><td>1.0</td></tr><tr><td>Top-p</td><td>0.85</td></tr><tr><td>Top-k</td><td>50</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Warm-up Ratio</td><td>0.05</td></tr><tr><td>Weight Decay</td><td>0.1</td></tr></table>

# C.3.3 Compute resources

LoRA-based supervised fine-tuning For the models in the 7-8 billion parameter range, this LoRA-based SFT training was conducted on a single A800 GPU and the training duration lasted around 15 hours or less for 40k samples (1 epoch).

# GRPO-based reinforcement learning

- 7-8 billion parameter model Training on 5k samples (1 epoch) required approximately 22 hours, utilizing five A800 GPUs, including resources for the deployment of the evaluator model.

- 3 billion parameter model The training process lasted approximately 18 hours for 5k samples (1 epoch) on four A800 GPUs, also including the evaluator model's deployment.

GPU usage Training a 7-8 billion parameter model with GRPO-based RL required approximately 22 hours, utilizing five A800 GPUs. This GPU allocation included resources for the deployment of the evaluator model. For a 3 billion parameter model, the training process lasted approximately 18 hours on four A800 GPUs, with this count also inclusive of the evaluator model's deployment. For the 32 billion parameter model, the LoRA-based training took approximately 22 hours using eight H20 GPUs with 141GB memory each, while the reward model was deployed on four A100 GPUs.

# C.4 Inference and evaluation

Besides trained models, we also evaluated proprietary large-scale models such as GPT-4o (20240806) [20] and Claude 3.5 Sonnet (20241022) [1], and open-source models that represent the current state of the art, including InternVL3-78B and Qwen2.5-VL-72B.

First, for evaluation on both GameQA and general vision benchmarks, the inference and evaluation configurations were unified across the original models and our trained models, detailed below.

For inference, the inference temperature parameter is set to 0.2, and the prompt is shown in Appendix H.3. We evaluated the generated answers using the LLM-as-a-judge approach [37], with Qwen2.5-72B-AWQ acting as the evaluator and the prompt for evaluation shown in H.4. To improve the reliability of the evaluation results, we introduced a series of engineering optimizations that ensured consistent and accurate assessments.

Second, We adopted different model evaluation strategies for different types of experiments to balance efficiency, accuracy, and the reliability of our conclusions, detailed below.

- Selecting the model checkpoint with best performance on the validation set for evaluation This approach is employed for the GRPO-based training experiments shown in Table 3, 5, 6 and the SFT experiments detailed in Appendix D.1. We saved 10 evenly spaced model checkpoints and evaluated the model that achieved the best performance on the validation set. The validation set was created by splitting $1\%$ of the training data. It was found that the selected model was generally the final model in the training process. This approach efficiently and directly reflects the ultimate performance of our methods in practical scenarios.   
- Evaluating the top three model checkpoints based on the validation set performance and averaging the evaluation results This is employed for the comparative training shown in Table 4 and the analytic experiments of Section 5.4, Appendix ?? and Appendix D.2. This approach was chosen because the performance differences are often subtle and susceptible to randomness in the training process. Averaging across multiple top checkpoints helps smooth out training noise, enhances the robustness and sensitivity of the evaluation, and more accurately reflects the differences under various experimental settings. This method also reduces the impact of chance results, making our experimental conclusions more reliable and representative.

# D More Experiments

# D.1 SFT experiments

As described in Appendix C, we performed additional supervised fine-tuning (SFT) experiments on four models. The results of these experiments are presented in Tables 9 and 10, and we found that:

The GRPO demonstrates an advantage over SFT in out-of-domain generalization, though SFT yields strong in-domain gains. When evaluating model performance, the effects of SFT and GRPO training methods showed clear differences across domains. Specifically, SFT training with the GameQA dataset led to substantial in-domain improvements: the InternVL2.5-8B and Qwen2.5-VL-7B models improved their average accuracy by $24.51\%$ and $22.59\%$ respectively on a test set covering 20 games, reaching final scores of $46.73\%$ and $48.37\%$ (Table 9). However, while SFT excels at improving in-domain performance, it can sometimes lead to performance degradation on general tasks as seen in 9, a phenomenon known as "catastrophic forgetting". Crucially, for training sets that belong to the mathematics domain, such as Geo-Multi (see Appendix C.2), models trained

Table 9 Evaluation results on GameQA benchmark. In-domain and out-of-domain game category results are shown. The percentage of performance improvements compared to the vanilla model is denoted by $(\uparrow)$ . Best performance per section is indicated in bold.   

<table><tr><td rowspan="2">Models</td><td rowspan="2">Avg. (↑)</td><td rowspan="2">3D Spatial Perc. &amp; Under.</td><td colspan="3">In Domain Games</td><td rowspan="2">Avg. (↑)</td><td rowspan="2">3D Spatial Perc. &amp; Under.</td><td colspan="3">Out of Domain Games</td></tr><tr><td>Pattern Recog. &amp; Matching</td><td>Multi-step Reasoning</td><td>Strategic Planning</td><td>Pattern Recog. &amp; Matching</td><td>Multi-step Reasoning</td><td>Strategic Planning</td></tr><tr><td>InternVL2.5-8B</td><td>22.22</td><td>20.39</td><td>17.18</td><td>25.34</td><td>25.97</td><td>20.05</td><td>20.80</td><td>22.45</td><td>18.88</td><td>18.07</td></tr><tr><td>+ GameQA (SFT)</td><td>46.73 (+24.51)</td><td>42.38</td><td>45.55</td><td>51.56</td><td>47.44</td><td>25.81 (+5.76)</td><td>24.40</td><td>24.69</td><td>25.90</td><td>28.25</td></tr><tr><td>+ GameQA (GRPO)</td><td>29.44 (+7.21)</td><td>26.74</td><td>26.05</td><td>29.51</td><td>35.44</td><td>23.87 (+3.82)</td><td>25.00</td><td>25.12</td><td>24.91</td><td>20.45</td></tr><tr><td>InternVL3-8B</td><td>26.51</td><td>22.53</td><td>21.91</td><td>30.18</td><td>31.43</td><td>27.64</td><td>29.60</td><td>27.44</td><td>29.62</td><td>23.91</td></tr><tr><td>+ GameQA (SFT)</td><td>50.72 (+24.21)</td><td>46.91</td><td>45.42</td><td>55.91</td><td>54.63</td><td>26.02 (-1.62)</td><td>24.20</td><td>14.80</td><td>32.03</td><td>33.05</td></tr><tr><td>+ GameQA (GRPO)</td><td>33.09 (+6.58)</td><td>27.94</td><td>28.52</td><td>36.81</td><td>39.07</td><td>28.80 (+1.15)</td><td>29.20</td><td>25.31</td><td>34.59</td><td>26.09</td></tr><tr><td>Qwen2.5-VL-7B</td><td>25.78</td><td>22.58</td><td>21.92</td><td>25.21</td><td>33.40</td><td>27.09</td><td>23.60</td><td>29.20</td><td>26.21</td><td>29.34</td></tr><tr><td>+ GameQA (SFT)</td><td>48.37 (+22.59)</td><td>42.05</td><td>42.82</td><td>58.66</td><td>49.96</td><td>29.27 (+2.18)</td><td>26.80</td><td>21.16</td><td>31.79</td><td>37.32</td></tr><tr><td>+ GameQA (GRPO)</td><td>32.12 (+6.34)</td><td>26.80</td><td>26.88</td><td>33.34</td><td>41.45</td><td>30.51 (+3.42)</td><td>27.10</td><td>31.56</td><td>31.24</td><td>32.13</td></tr><tr><td>LLVA-OV-7B</td><td>21.79</td><td>18.84</td><td>19.69</td><td>21.24</td><td>27.38</td><td>20.39</td><td>21.00</td><td>20.19</td><td>20.13</td><td>20.27</td></tr><tr><td>+ GameQA (SFT)</td><td>45.40 (+23.61)</td><td>38.50</td><td>39.47</td><td>54.78</td><td>48.84</td><td>22.74 (+2.34)</td><td>22.00</td><td>17.17</td><td>25.16</td><td>26.63</td></tr><tr><td>+ GameQA (GRPO)</td><td>33.49 (+11.70)</td><td>29.87</td><td>31.10</td><td>30.96</td><td>42.03</td><td>23.34 (+2.95)</td><td>27.20</td><td>20.05</td><td>23.55</td><td>22.57</td></tr></table>

Table 10 Evaluation results on general vision benchmarks. The percentage of performance improvements compared to the vanilla model is denoted by $(\uparrow)$ . Best performance per section is indicated in bold.   

<table><tr><td>Models</td><td>Avg.(↑)</td><td>MathVista</td><td>MathVerse</td><td>MMBench</td><td>MMMU</td><td>CharXiv</td><td>MathVision</td><td>MMMU-Pro</td></tr><tr><td>InternVL2.5-8B</td><td>45.89</td><td>57.50</td><td>36.04</td><td>81.93</td><td>47.96</td><td>31.70</td><td>28.87</td><td>37.25</td></tr><tr><td>+ GameQA (SFT)</td><td>45.06 (-0.83)</td><td>58.10</td><td>35.79</td><td>82.87</td><td>48.07</td><td>31.20</td><td>22.80</td><td>36.56</td></tr><tr><td>+ Geo-Multi (SFT)</td><td>43.84(-2.05)</td><td>53.60</td><td>36.90</td><td>82.80</td><td>43.17</td><td>30.40</td><td>27.80</td><td>32.22</td></tr><tr><td>+ GameQA (GRPO)</td><td>47.91 (+2.02)</td><td>61.70</td><td>37.11</td><td>83.87</td><td>50.06</td><td>32.00</td><td>31.93</td><td>38.69</td></tr><tr><td>InternVL3-8B</td><td>54.48</td><td>69.10</td><td>50.10</td><td>86.00</td><td>57.88</td><td>39.10</td><td>35.33</td><td>43.84</td></tr><tr><td>+ GameQA (SFT)</td><td>49.66(-4.82)</td><td>63.20</td><td>43.30</td><td>84.53</td><td>53.44</td><td>32.90</td><td>29.60</td><td>40.64</td></tr><tr><td>+ Geo-Multi (SFT)</td><td>48.52 (-1.42)</td><td>62.60</td><td>40.96</td><td>82.87</td><td>48.54</td><td>37.80</td><td>29.80</td><td>37.06</td></tr><tr><td>+ GameQA (GRPO)</td><td>55.88 (+1.40)</td><td>73.00</td><td>50.71</td><td>86.20</td><td>58.34</td><td>39.90</td><td>37.93</td><td>45.10</td></tr><tr><td>Qwen2.5-VL-7B</td><td>49.94</td><td>66.80</td><td>45.08</td><td>83.67</td><td>49.01</td><td>37.70</td><td>30.80</td><td>36.49</td></tr><tr><td>+ GameQA (SFT)</td><td>47.26 (-2.72)</td><td>63.00</td><td>37.31</td><td>83.07</td><td>47.49</td><td>38.60</td><td>25.73</td><td>35.62</td></tr><tr><td>+ Geo-Multi (SFT)</td><td>48.52 (-1.42)</td><td>62.60</td><td>40.96</td><td>82.87</td><td>48.54</td><td>37.80</td><td>29.80</td><td>37.06</td></tr><tr><td>+ GameQA (GRPO)</td><td>52.27 (+2.33)</td><td>68.20</td><td>47.97</td><td>83.53</td><td>50.53</td><td>42.70</td><td>33.07</td><td>39.89</td></tr><tr><td>LLaVA-OV-7B</td><td>41.23</td><td>55.60</td><td>33.05</td><td>81.13</td><td>41.07</td><td>27.10</td><td>23.40</td><td>27.26</td></tr><tr><td>+ GameQA (SFT)</td><td>35.83 (-5.40)</td><td>45.90</td><td>25.99</td><td>80.27</td><td>32.21</td><td>20.50</td><td>20.47</td><td>25.44</td></tr><tr><td>+ Geo-Multi (SFT)</td><td>40.35 (-0.88)</td><td>52.50</td><td>31.57</td><td>82.40</td><td>41.31</td><td>25.90</td><td>22.60</td><td>26.19</td></tr><tr><td>+ GameQA (GRPO)</td><td>42.27 (+1.04)</td><td>58.20</td><td>34.92</td><td>82.53</td><td>41.31</td><td>27.30</td><td>23.07</td><td>28.58</td></tr></table>

on them can still exhibit a decline in general capabilities. This may reflect the tendency of the SFT method to overfit the model to specific domain data [5].

In contrast, when training on GameQA using the GRPO, models not only successfully avoided performance degradation on general tasks but also generally achieved performance improvements across multiple general visual benchmarks (Table 3). A typical example is the Qwen2.5-VL-7B model, which, after GRPO training, showed enhanced performance on challenging benchmarks such as MathVista, MathVerse, and CharXiv, reaching a level comparable to larger-scale models like InternVL2.5-38B.

Pure RL outperforms SFT-then-RL in out-of-domain generalization. To validate our choice of directly applying RL, we compare our approach to a two-stage SFT-then-RL pipeline on Qwen2.5-VL-7B. As shown in Table 11, while the two-stage pipeline yields strong performance on in-domain tasks, it leads to a significant performance drop of $-2.5\%$ on general vision benchmarks. This suggests that full-parameter SFT on a narrow, specialized domain like GameQA can cause catastrophic forgetting. In contrast, pure RL better preserves and enhances out-of-domain generalization capabilities, achieving a $+2.33\%$ improvement.

Table 11 Comparison of training pipelines on Qwen2.5-VL-7B. SFT is performed on ${20}\mathrm{k}$ samples,followed by GRPO on $5\mathrm{k}$ . Performance on general benchmarks is shown as a relative change from the baseline.   

<table><tr><td>Training Stage</td><td>In-domain Games</td><td>Out-of-domain Games</td><td>General Benchmarks (Avg. Change)</td></tr><tr><td>Baseline (Before SFT)</td><td>25.89</td><td>26.92</td><td>0.0%</td></tr><tr><td>SFT</td><td>56.21</td><td>30.74</td><td>-3.9%</td></tr><tr><td>SFT-then-RL</td><td>58.08</td><td>31.96</td><td>-2.5%</td></tr><tr><td>Pure RL (Our Method)</td><td>32.12</td><td>30.51</td><td>+2.33%</td></tr></table>

# D.2 Training data difficulty experiment

To analyze how reasoning difficulty impacts generalization, we categorized tasks into Easy (55-85% baseline accuracy), Medium (30-55%), and Hard (5-30%). We trained Qwen2.5-VL-7B on 5k samples from different difficulty combinations. Table 12 shows that training on a diverse mix of difficulties yields the best generalization. The model trained on "Easy+Medium+Hard" samples achieves the highest average score (52.74), outperforming models trained on simpler subsets. This confirms that the complex reasoning patterns in GameQA are crucial for improving general reasoning.

Table 12 Impact of training data difficulty on generalization. Results are averaged over the top three checkpoints. The baseline score is 49.94.   

<table><tr><td>Training Data (5k)</td><td>Average</td><td>MathVista</td><td>MathVerse</td><td>MMBench</td><td>MMMU</td><td>CharXiv</td><td>MathVision</td><td>MMMU-Pro</td></tr><tr><td>Baseline</td><td>49.94</td><td>66.80</td><td>45.08</td><td>83.67</td><td>49.01</td><td>37.70</td><td>30.80</td><td>36.49</td></tr><tr><td>Easy</td><td>52.40</td><td>67.87</td><td>47.93</td><td>83.91</td><td>51.58</td><td>41.77</td><td>33.73</td><td>40.03</td></tr><tr><td>Medium</td><td>51.95</td><td>67.30</td><td>48.10</td><td>83.64</td><td>50.53</td><td>40.83</td><td>33.96</td><td>39.30</td></tr><tr><td>Hard</td><td>52.07</td><td>67.57</td><td>47.53</td><td>83.58</td><td>51.03</td><td>40.97</td><td>34.58</td><td>39.26</td></tr><tr><td>Easy+Medium</td><td>52.45</td><td>68.60</td><td>48.07</td><td>83.42</td><td>51.30</td><td>41.40</td><td>34.86</td><td>39.49</td></tr><tr><td>Easy+Medium+Hard</td><td>52.74</td><td>68.33</td><td>48.68</td><td>83.93</td><td>52.36</td><td>40.73</td><td>34.31</td><td>40.81</td></tr></table>

# D.3 Data scale experiment

We extended our data scaling experiment up to 20k samples. While individual checkpoints exhibit fluctuations common in RL, a "binned averaging" analysis (Table 13) reveals a clear, monotonic positive trend. By averaging performance across training stages, we smooth out short-term noise and observe a stable improvement trajectory. The average performance on general benchmarks steadily increases from +2.31 in the early stage to +3.02 in the late stage, confirming that model performance continues to improve with more data from GameQA.

Table 13 Binned averaging analysis of data scaling effect on Qwen2.5-VL-7B. Performance gains over baseline are shown in parentheses.   

<table><tr><td>Training Stage</td><td>Sample Range</td><td>General Vision Benchmarks</td><td>In-domain Games</td></tr><tr><td>Baseline Model</td><td>0k</td><td>49.94</td><td>25.78</td></tr><tr><td>Early Stage Training</td><td>1k-5k</td><td>52.25 (+2.31)</td><td>27.68 (+1.90)</td></tr><tr><td>Mid Stage Training</td><td>6k-15k</td><td>52.57 (+2.63)</td><td>34.97 (+9.19)</td></tr><tr><td>Late Stage Training</td><td>16k-20k</td><td>52.96 (+3.02)</td><td>38.69 (+12.91)</td></tr></table>

# E Approach details

# E.1 Task category

Target Perception Task focuses on visual perception and basic state awareness. State Prediction Task, building directly on the perceptual capacities, needs predictions about state transitions. Strategy Optimization Task then needs both perceptual and predictive capacities to find optimal solutions. This progressive structure helps organize reasoning skills from simple to complex.

The conceptual outline for each category, using Sokoban as an example, is as follows:

- Target Perception Task: Queries static information within the game state. For instance, questions ask about the position or number of boxes, and the answers list the specific positions of each box by directly inspecting the current state.   
- State Prediction Task: Infers state changes following actions. For instance, questions predict the player's position after a sequence of moves. Answers are derived by analyzing the initial state, simulating the execution of each step, recording the resulting state changes, and thus determining the final player position.   
- Strategy Optimization Task: Aims to find optimal solutions. For instance, questions ask for the shortest path to push a specific box to its designated target. The answer is derived by first analyzing the initial state to determine the optimal route for moving the box to its target, and then simulating the execution of this optimal action sequence.

# E.2 Selection criteria of 30 games in GameQA

We selected these 30 games based on the following criteria.

- Ability coverage The games need to cover a diverse range of reasoning abilities, including 3D Spatial Perception and Understanding, Pattern Recognition and Matching, Multi-step Reasoning, and Strategic Planning.   
- Code simplicity The code should be easy to construct, meaning they are simple enough to be programmed by an LLM, or are open-sourced.   
- Static game They should be static or can be transformed into a static state, so that problems can be solved from a static image.

# E.3 Time spent on the main steps of Code2Logic for each game

Figure 11 illustrates the estimated time spent on implementing the main steps of the Code2Logic approach across all the 30 games in the GameQA dataset. The time ranges from a minimum of 4 hours to a maximum of 12 hours, with an average of 7.5 hours per game. Given that all annotators were undergraduates and that ten of them completed only a single game, the time required for a more experienced annotator could be significantly less than this average.

This average time investment is relatively cost-effective and appears highly acceptable, especially considering that once the code data engine is built, it can generate an unlimited number of data samples for training and evaluation purposes.

![](images/c161d8fedefd811999b30eb16508b9116c0ab55064fdafb8bad274d22b1afc50.jpg)  
Figure 11 Estimated time (in hours) spent on implementing the main steps of Code2Logic across different games in the GameQA dataset, with an overall average of 7.5 hours per game.

# E.4 Games generated using external open-source code

This section lists the games whose code is generated based on open-source code, as referenced in Section 2.1.

Spider Solitaire (Open-source code URL: https://github.com/rdasxy/spider_solitaire)

Based on an original Python implementation of Spider Solitaire, our code reused its core rules and GUI. And we simplified the game to a single suit (Spades) to reduce complexity, enriched initial setups through LLM-implemented random deals, and adapted the original game rules into detailed instructions.

Klondike (Open-source code URL: https://github.com/milorb/klondike)

Based on the original open-source Klondike Solitaire project built on Pygame<sup>3</sup>, we adapted the code using an LLM into an automated dataset generation tool. It reused the core game engine and introduced a method for generating diverse random initializations.

Space Invaders (Open-source code URL: https://github.com/leerob/space-invaders)

Based on the original Space Invaders game built with Pygame, we utilized its core elements and visual assets to generate static game scenes. Using an LLM, we converted the dynamic game into static scene snapshots for dataset generation.

# F More information about the GameQA Dataset

# F.1 The GameQA dataset statistics

The full GameQA train and test set statistics table is shown in Table 14.

# F.2 Definition of each category

3D Spatial Perception and Reasoning Game of this type involves the ability to perceive, plan, and reason in 3D space to complete tasks such as navigation and spatial transformation. For example, in 3D Reconstruction, the task is to reconstruct the stacking arrangement of 3D cubes from a side view. Solving this task requires 3D spatial reasoning to establish the relationship between the 2D view and the 3D view.

Pattern Recognition and Matching Game of this type requires capability on discerning and matching visual patterns related to object shapes, colors, combinations, and other regularities. For example, in Tangram, the

task is to identify which piece can fill the empty space. Solving this task requires recognizing the shape of void and matching with given pieces.

Multi-step Reasoning Game of this type features multi-step reasoning and iteratively applying rules to reach the solution. For example, in Selenium, the task is to infer which color should fill the empty space to ensure that no colors are repeated in the same row, column, or 3x3 grid. The "no repetition" rule needs to be applied repeatedly to deduct the correct color of a cell.

Strategic Planning Game of this type requires planning the optimal solution in optimization problems. For example, in Sokoban, the task is to plan the shortest path for pushing a box from the starting point to the target location.

# F.3 Data augmentation

To prevent model overfitting to specific reasoning patterns, we employed an LLM-based reasoning paraphrase strategy using InternVL2.5-78B. Based on initial experiments showing that visual input could lead to errors due to the model's insufficient visual capabilities, we provided it only with textual information, namely the question, original answer

Table 14 GameQA train and test set statistics summary. All lengths are calculated by words.   

<table><tr><td>Statistic Category</td><td>Train Set</td><td>Test Set</td></tr><tr><td colspan="3">Overall Counts</td></tr><tr><td>Total Games</td><td>20</td><td>30</td></tr><tr><td>Total Tasks</td><td>102</td><td>158</td></tr><tr><td>Total Questions</td><td>126,760</td><td>15,047</td></tr><tr><td colspan="3">Image Statistics</td></tr><tr><td>Unique Images</td><td>74,620</td><td>8,620</td></tr><tr><td>Avg. Image Width (px)</td><td>511.00</td><td>504.10</td></tr><tr><td>Avg. Image Height (px)</td><td>475.73</td><td>468.98</td></tr><tr><td colspan="3">Question Characteristics</td></tr><tr><td>Avg. Question Length</td><td>275.27</td><td>272.43</td></tr><tr><td>Avg. Analysis Length</td><td>106.85</td><td>144.89</td></tr><tr><td>- After Augmentation</td><td>300.79</td><td>-</td></tr><tr><td>Multiple Choice Questions</td><td>86,520</td><td>10,518</td></tr><tr><td>Avg. Choices for MCQs</td><td>7.10</td><td>7.05</td></tr><tr><td>Fill-in-the-Blank</td><td>40,240</td><td>4,529</td></tr></table>

process and final answer, to rewrite the answer process. This approach enriches linguistic style and logical expression diversity while maintaining semantic consistency.

# F.4 Data quality assurance

To ensure the high quality and reliability of our synthetic dataset, we implemented a quality assurance process, consisting of four stages:

1. Human Inspection: STEM students inspected initial samples to ensure logical correctness, clarity and completeness of questions, images and reasoning steps.   
2. LLMs Check: Fed samples to GPT-4o and Claude-3.5 Sonnet to ensure model comprehensibility, identifying necessary refinements in the samples.   
3. Post-Augmentation Verification: Manually verified reasoning accuracy in a random subset of augmented data.   
4. Automated Data Filtering: Removed samples based on length, high repetition (>70% 4-gram overlap), or wrong answers, reducing the set from $\tilde{150k}$ to 126,760.

# G Details on Sokoban task synthesis

# G.1 Sokoban QA templates

Here we provide templates used for generating Sokoban puzzle samples, including the three problem types: Target Perception, State Prediction, and Strategic Optimization.

Target perception QA template The Target Perception template (Table 15) is used to generate questions about identifying the current position of game elements.

State prediction QA template The State Prediction template (Table 16) is used to generate questions about predicting the final position of the player after a sequence of moves.

```txt
"question": "This is a Sokoban puzzle where Cartoon people is player, green X is target, brown box with X is box to push, brown tiles are walls, and light brown areas are movable spaces. The coordinates (x, y) in this puzzle represent the matrix format. What is the current position of the <object> (row, column)?\nOptions: \n[1] <option_1>\n[2] <option_2>\n[3] <option_3>\n[4] <option_4>\n[5] <option_5>\n[6] <option_6>\n[7] <option_7>\n[8] <option_8>", 
```

```txt
"answer": <number>, 
```

```txt
"analysis": "Player position: <pos>\nBoxes positions: <pos>\nTarget positions: <pos>\nThe player is currently at position <pos>.\nSo the answer is <answer>. The option number is <number>.", 
```

Table 16 State prediction QA template

```txt
"question": "This is a Sokoban puzzle where Cartoon people is player, green X is target, brown box with X is box to push, brown tiles are walls, and light brown areas are movable spaces. The coordinates (x, y) in this puzzle represent the matrix format. If the player makes these moves: <mov_seq>, where will player end up? \n\NOptions: \n[1] <option_1> \n[2] < option_2> \n[3] <option_3> \n[4] <option_4> \n[5] <option_5> \n [6] <option_6> \n[7] <option_7> \n[8] <option_8>", "answer": <number>, "analysis": "Player position: <pos> \nMove 1 - <dir>: Player moves from <pos> to <pos> \nMove 2 - <dir>: Player moves from <pos> to <pos> \n... \nFinal position: <pos> \nThe option number is < number>", 
```

Strategic optimization QA template The Strategic Optimization template (Table 17) is used to generate questions about finding the optimal sequence of moves between positions.

# G.2 Supplementary prompt

As mentioned in Section 2.2. LLMs can not only refine the human-designed templates, but also design new questions and QA templates with the example prompt provided below. We will conduct a careful manual review of the quality of tasks generated by the LLM and make selections, even if we have already prompted to generate diverse and meaningful QA pairs.

# Prompt for designing new questions and the corresponding QA templates

Generate Game QA Derivative Templates Based on the provided basic QA template for the Sokoban game, please design more question-answering template variations. The reference file already includes three basic template categories:

"question": "This is a Sokoban puzzle where Cartoon people is player, green X is target, brown box with X is box to push, brown tiles are walls, and light brown areas are movable spaces. The coordinates (x, y) in this puzzle represent the matrix format. Treat the boxes as walls, What is the shortest sequence of moves for human to move himself from position $<pos>$ to position $<pos>$ ?\n\NoOptions:\n[1] <option_1>\n[2] <option_2>\n[3] <option_3>\n[4] <option_4>\n[5] <option_5>\n[6] <option_6>\n[7] <option_7>\n[8] <option_8>,"

```txt
"answer": <answer_number>, 
```

```txt
"analysis": "Player position: <pos>\nBoxes positions: <pos>\nTarget positions: <pos>\nStart position: <pos>\nEnd position: <pos>\nOptimal move sequence: <mov_seq>\nMove 1 - <dir>: Player moves from <pos> to <pos>\nMove 2 - <dir>: Player moves from <pos> to <pos>\nFinal position: <pos>\nSo the answer is <answer>. The option number is <number>.", 
```

1. State Prediction - Predict the player's position after a move.   
2. Target Perception - Identify the current positions of game elements.   
3. Strategy Optimization - Find the optimal movement path.

Please design 3-5 innovative derivative templates for each category, ensuring the new templates:

- Maintain consistency with the original JSON format.   
- Cover different reasoning difficulties and complexities.   
- Test different cognitive and reasoning abilities.

When designing, please follow this reasoning hierarchy:

- Level 1: Target Perception QA - Focus on basic visual recognition and state understanding (e.g., "Where is the box?").   
- Level 2: State Prediction QA - Focus on state changes and transition reasoning (e.g., "After performing these moves, where will the player be?").   
- Level 3: Strategy Optimization QA - Focus on finding the optimal solution (e.g., "What is the minimum number of moves to push the box to the target?").

For each new template, please provide:

1. Template name and type classification.   
2. Complete JSON template structure (including all necessary placeholders).   
3. A brief description explaining the specific abilities tested by the template.   
4. Placeholder filling examples (how to generate specific question instances).

Please ensure your template designs can generate diverse and meaningful Q&A pairs based on the game state and maintain consistency with the original template structure.

# H Prompts

# H.1 Data augmentation

Below is the prompt used to perform the LLM-based reasoning paraphrase strategy detailed in Appendix F.3, with visual information not provided for the model.

# Prompt for data augmentation

Question: {query} (the question generated by the data engine)   
Ground Truth: {response} (the answer with reasoning process generated by the data engine)

Based on the above question and the provided ground truth, the current process of providing the answer is overly mechanical and simplistic. Please provide detailed reasoning steps based on the content of the question and the reasoning steps in the Ground Truth. The reasoning steps should be detailed, logical, and consistent with the Ground Truth.

Additionally, before starting the reasoning, emphasize: "I will carefully analyze the question and the image and provide detailed reasoning steps." Do not include statements such as "This matches the provided Ground Truth" or similar expressions in your response.

Please follow the above requirements to provide a detailed analysis, reasoning, and answer.

# H.2 Training

# System prompt for the model being trained in GRPO

Please carefully observe the image, thoroughly understand the conditions provided in the question, use logical reasoning to arrive at the result, and reflect on and verify the reasoning process to ensure the accuracy of the answer. Finally, provide the correct answer.

# Prompt for the LLM-based judging module in GRPO

# System prompt:

Compare the ground truth with the prediction from AI model and determine if the prediction is correct. The question is about an image, which we have not given here. You need to determine whether the model's prediction is consistent with the ground truth. No points will be awarded for wrong answers, over answers or under answers. There are times when the answer may have a different form of expression and some variation is acceptable.

# User instruction prompt:

Ground Truth: The correct answer is {answer}.

(For multiple choice question: The correct option is {answer}: {option_content}.)

```python
## Prediction: {simplified_prediction}

Correctness: (Yes or No)

# H.3 Inference

# Prompt for inference

{query} Let's think step by step. Please analyze the question carefully and follow these requirements:

Provide detailed step-by-step reasoning,

Show all your work and calculations,

End your response with one of these formats:

1. For choice questions: 'The answer is [option]'   
2. For other questions: 'The answer is [final answer]'

The final answer line must be on its own line at the very end of your response.

# H.4 Evaluation

# Prompt for evaluation

# System prompt:

Compare the ground truth with the prediction from AI model and determine if the prediction is correct.

The question is about an image, which we have not given here. You need to determine whether the model's prediction is consistent with the ground truth. No points will be awarded for wrong answers, over answers or under answers.

The reasoning process in the prediction does not need to be considered too much, you only need to determine if the final answer is consistent. There are times when the answer may have a different form of expression and some variation is acceptable.

# User instruction prompt:

Question: {simplified(question)}

Ground Truth: The correct answer is {answer}.

(For multiple-choice question: The correct option is {answer}: {option_content}.)

```python
## Prediction: {simplified_prediction}

You need to determine whether the model's prediction is consistent with the ground truth. Output only:

Correctness: (Yes or No)

# I Case study on model performance on GameQA

We present GPT-4o case studies on different GameQA games below, to showcase deficiencies of GPT-4o on visual perception and reasoning. Our qualitative analysis of model performance across four game types reveals common behaviors and challenges.

3D Spatial Perception and Understanding Visual language models exhibit significant limitations in 3D spatial reasoning games and score lowest among the four game categories. For example, in Figure 12, GPT-4o struggled on ordering green cubes by their height (Z-coordinate), recognizing cube 1, that is closer to top of the image, as having higher Z-coordinate than cube 2. Moreover, in Figure 13, GPT-4o failed to identify the position of the only voxel in the image. In Figure 14, GPT-4o appeared to be unable to count how many cubes are on each edge of the cuboid, which is generally easy for humans.

Pattern Recognition and Matching In this game category, we find that models faced difficulties in identifying patterns and locating objects. This was particularly challenging with non-grid layouts or images without row

![](images/86944a3eaf3f6c2834b0a22127b3075cc9b5d895194bbc1352eb58f9740b92de.jpg)

Question: What is the correct height relationship between the three numbered points?

Answer: $1 = 2 < 3$

GPT-4o Response: Point 1 is on top of a cube that is one level above the blue starting cube. Point 2 is on the same level as the blue starting cube ... Therefore, the height relationship is: $2 < 1 < 3$ .

Error Analysis: Point 1 is at same height as point 2.

![](images/9790d8a0e48d5c4a93dabb5ac5de2a1897854809cb6388439ea21be51bdc40aa.jpg)  
Figure 12 GPT-4o 3D Maze case study   
3D Voxel Reconstruction Game

![](images/e27bc9ac3b6681a369856ec419ce6f35ef0b67f0b62aab88ef278db2af497379.jpg)

![](images/6eb4aa30d4548c4abc859fa9ae8f7d5b479b5151b444b47e3a9adfb05c8c4e36.jpg)

Question: Goal: Reconstruct a 3D structure by adding voxels to match given projections. Position $(x,y,z)$ ranges from 1 to 3, with $(1,1,1)$ at front-left-bottom. All voxels must be connected face-to-face. New voxels can only be placed adjacent to existing ones. Choose the correct sequence from the options below.

# Options:

1: Add voxels at positions: $[(1,3,1),(3,3,1)]$

Answer: 1

GPT-4o Response: To solve this puzzle, we need to add voxels to the current structure such that the projections match the given Front View (Y-Z) and Side View (X-Z).

Let's analyze the projections:

1.Front View (Y-Z Plane):

1. The projection shows a filled cell at (2, 1), indicating there is at least one voxel along the line of sight at $y = 2$ and $z = 1$ .   
2. The projection shows a filled cell at (3, 1), indicating there is at least one voxel along the line of sight at $y = 3$ and $z = 1$ .

2.Side View (X-Z Plane):

1. The projection shows filled cells at (1, 1), (2, 1), and (3, 1), indicating there is at least one voxel along the line of sight at $x = 1$ , $x = 2$ , and $x = 3$ , all at $z = 1$ .

Given the current structure, we have a voxel at $(2, 2, 1)$ . We need to add two more voxels to match both projections ...

Error Analysis: GPT-4o gives wrong position of the existing voxel.

Figure 13 GPT-4o 3D Reconstruction case study

![](images/9daad563e8860722fb6df54f3157fcc9b238e986fa3566c115bdf8cf8155dcba.jpg)  
Figure 14 GPT-4o Minecraft case study

Question: How many cubes are there in total in the scene?

Answer: 48

GPT-4o Response: The scene shows a cube composed of smaller cubes. By counting the cubes along one edge, we can determine the dimensions of the larger cube. There are 5 cubes along each edge. Therefore, the total number of smaller cubes in the scene is calculated by cubing the number of cubes along one edge: $5 \times 5 \times 5 = 125$ . So, there are 125 cubes in total in the scene.

Ground Truth: Looking at the top of it, we know that the length and the width are 4 and 4 respectively. Besides, we can see that the height is 3. Therefore, the total number of cubes is $4 \times 4 \times 3 = 48$ . The answer is 48.

Error Analysis: GPT-4o counts number of cubes in each edge wrong.

and column indicators. As seen in Figure 15, GPT-4o has poor performance in circular Zuma grid, claiming to find 3 pairs of same color marbles, but none is correct. GPT-4o also meets problems in the card games. In Figure 16, GPT-4o fails to understand "top of a pile" appears lower in the image. In Figure 17, GPT-4o misreads the rank of the card.

![](images/c4de0e1e64867c988355607d7eeddc8a90e2993268da87913e8dd427e1f895c5.jpg)

Question: How many marble groups of two or more same-colored marbles are there at the up-right side of the frog?

Answer: 0

GPT-4o Response: Starting from the top, the first group in the 'up-right' region is a group of two yellow marbles. Moving clockwise, the next group is a group of two blue marbles.

Continuing further, there is a group of two red marbles ... there are a total of 3 groups.

Error Analysis: None of the three groups GPT-4o claims is correct.

Figure 15 GPT-4o Zuma case study. The red rectangle is added to highlight the area referred to in the question and does not exist in original image.

![](images/98f02a59912a1d8c9d8eba82a03eebed187176166d5c80c334df56274f4549cd.jpg)  
Figure 16 GPT-4o Freecell case study

Question: In this FreeCell game, we have 8 cascade piles at sum, and their indexes are [0, 1, 2, 3, 4, 5, 6, 7]. In FreeCell, cards can be moved according to specific rules: A card can be moved to a free cell if available, stacked in descending order alternating colors in cascade piles, or placed in foundation piles starting from Ace. Now, find the 3rd card from the top of cascade pile 1. Options are as follows:

1.(Diamond, 5) 2.(Club, 1) 3.(Spade, 6) 4.(Heart, 2) 5.(Club, 4) 6.(Heart, K) 7.(Heart, A) 8.(Spade, 9)

Answer: 3

GPT-4o Response: To find the 3rd card from the top of cascade pile 1, we need to look at the cards in that pile. The cards in cascade pile 1, from top to bottom, are: 1. 5 of Hearts 2. 10 of Hearts 3. Ace of Clubs. Therefore, the 3rd card from the top of cascade pile 1 is the Ace of Clubs. The correct option is: 2. (Club, 1)

Ground Truth: The selected pile1 contains (from top to bottom) the following cards: (Spade,5), (Heart,3), (Spade,6), (Spade,Q), (Club,A), (Diamond,10), (Heart,5). The 3rd card from the top is (Spade,6).

Error Analysis: GPT-4o does not understand top of a pile appears lowest in image.

![](images/a146a8bb5ccc8ed33ed703d76e0bea7481f6217bf4bae8d957eddc26d8d74478.jpg)  
Question: Piles are numbered from left to right starting with `0` . Cards in a pile are numbered from on-screen top to bottom starting with `0` . Cards can be moved to another pile as long as they form a descending sequence of the same suit. The cards within each waste pile are also numbered starting from the bottom card. What will happen if I want to move the number 6 card of pile 6 to pile 3? Options: A. The move will be successful, and the cards will be in descending order, following the rules of movement. B. The move cannot be made because this card is face-down and its value is unknown. C. The move cannot be made because ... Answer: A   
GPT-4o Response: In the image provided: Waste pile 6 has a visible card, which is a Jack (J) of spades. Waste pile 3 has a visible card, which is a King (K) of spades. The number 6 card in pile 6 would be the sixth card from the bottom of the pile. However, in the image, pile 6 only has one face-up card, the Jack of spades, and the rest are face-down. Therefore, the sixth card from the bottom is not visible and is face-down. Given this information, the correct option is: B. The move cannot be made because this card is face-down and its value is unknown.   
Ground Truth: Source pile 6 is the 7th pile from the left, and destination pile 3 is the 4th pile from the left. The number 6 card in pile 6 is the 7th card, which is Queen of Spade. The top card of pile 3 is King of Spade. So moving the Queen of Spade from pile 6 to pile 3 will be successful as it forms a valid descending sequence.   
Error Analysis: GPT-4o recognizes the visible card in pile 6 incorrectly, and misread the meaning of number 6 card.   
Figure 17 GPT-4o Spider Solitaire case study

Multi-step Reasoning Models have substantial deficiencies in multi-step reasoning. When a rule needs to be applied multiple times recursively, models are prone to repeat the initial application instead of performing subsequent reasoning steps. In Figure 19, GPT-4o repeats it finds prefixes of the target word "JUNKMAN" in options 2, 7 and 8, but actually the three options are far from the word "JUNKMAN". Another common flaw is failure to track game state changes. For example, in 2D Turing Machine game (Appendix J.3.5), models frequently forgot the symbol on a cell that had been modified in a previous step, still using the unchanged symbol to deduce the next move. A similar phenomenon was also observed in Langton's Ant game (Appendix J.3.3), where the model might not follow the new color of a certain cell that it had correctly recognized, still following the original cell color in the input image.

Strategic Planning These games show that models lack certain abilities to find the best strategy. They lack the human-like insight to prune unpromising choices, and are also unable to conduct large-scale search or traversal, resulting in poor performance. In the Sokoban game case (Figure 20), for example, the number of moves needed is 27, a relatively big number. GPT-4o directly states the answer is 29, without conducting any effective analysis. Irrational mistakes that seems more unreasonable also exist. As shown in Figure 21, GPT-4o states that after the first step, body of snake includes (0, 3) and new head position is (0, 3), not realizing that this overlap means the snake has hit its head. This reveals certain deficiencies in models' fundamental reasoning and spatial imagination skills.

In addition, it's worth mentioning that a bit counter-intuitively, models actually face major difficulties in visual perception under various game scenes, as shown in Figure 22, 18, 23 and 24, constraining their final performances.

# Minesweeper Board

![](images/5db4dc58594e9ff241f375fce88059c2210c8eca3ec6dd697bce589a2fb0bca0.jpg)

Question: This is a Minesweeper game. The size of the chessboard is 6x6, and there are a total of 8 mines hidden on the board. The numbers on the board indicate how many mines are adjacent to that cell, including diagonals. Cells marked with "F" (flagged) are identified as potential locations of mines based on logical deduction or prior knowledge.

Question: How many mines are left to be found?

Answer: 6

GPT-4o Response: To determine how many mines are left to be found, we need to analyze the board and the flagged cells. 1. Count the flagged cells: There are three flagged cells on the board: (0,5), (1,0), and (0,0). 2. Total mines: The problem states there are a total of 8 mines on the board. Total mines = 8 Flagged cells = 3 Mines left to be found = Total mines - Flagged cells = 8 - 3 = 5

Ground Truth: There are a total of 8 mines hidden on the board. By counting the number of F-marked cells (a total of 2), we can determine the remaining mines: 6

Error Analysis: GPT-40 incorrectly states $(0,0)$ is flagged.

![](images/3838ba6231ad6fbd2d30ef4ae01af4e436b3fed50cf24c0b8d1b7e1f7561faba.jpg)  
Figure 18 GPT-4o Minesweeper case study   
Option 2 Option 8 Option 7

Question: Find the word "JUNKMAN" in the grid. Where does it start and in which direction does it go? Options: ...

Answer: Row 7, Column 4, Direction: up

GPT-4o Response: ... Option 2: Row 1, Column 2, Direction: down. Starting at $(1,2) = \mathrm{J}'$ , $(2,2) = \mathrm{U}'$ , $(3,2) = \mathrm{N}'$ , $(4,2) = \mathrm{K}'$ , $(5,2) = \mathrm{M}'$ , $(6,2) = \mathrm{A}'$ , $(7,2) = \mathrm{N}'$ - This spells 'JUNKMAN', so this option is correct.

... Option 7: Row 1, Column 2, Direction: diagonal-right-down. Starting at $(1,2) = \mathrm{J}'$ , $(2,3) = \mathrm{U}'$ , $(3,4) = \mathrm{N}'$ , $(4,5) = \mathrm{K}'$ , $(5,6) = \mathrm{M}'$ , $(6,7) = \mathrm{A}'$ ...

Option 8: Row 7, Column 4, Direction: diagonal-left-up - Starting at $(7,4) = \mathrm{J}'$ - Diagonal-left-up: $(6,3) = \mathrm{U}'$ , $(5,2) = \mathrm{N}'$ , $(4,1) = \mathrm{K}'$ ...

Error Analysis: GPT-4o is hallucinating that the three options listed have letters similar to "JUNKMAN".

Figure 19 GPT-4o Word Search case study. Red rectangles and "Option X" texts are added to highlight the options and do not exist in original image.

![](images/1903bd52c6df7f78e0d51db21f3881e670a748427fee630b117d20f4d2c75eaa.jpg)  
Figure 20 GPT-4o Sokoban case study

Question: What is the minimum number of moves needed to solve this puzzle?

Answer: 27

GPT-4o Response: To solve this Sokoban puzzle, the player needs to push the box onto the target. The player must navigate around the walls and push the box in a way that it reaches the target with the minimum number of moves.

After analyzing the puzzle, the minimum number of moves needed to solve it is 29.

Error Analysis: GPT-4o states an answer without analyzing.

# J Details on the 30 games and more example data samples in the GameQA dataset

A total of 10 games are introduced below in detail, with detailed question information and specific example QAs provided, while the others 20 games in brief.

![](images/ae0194c4d6b8228b63f719737614803022b1275dc28ccee9b6f935d2af17e8b6.jpg)

Question: This is a Snake game. The yellow block is the head of the snake. The blue block is the body of the snake. The red block is the food. The coordinates $(x, y)$ in the grid represent the matrix format, where $x$ is the row index and $y$ is the column index. The origin $(0, 0)$ is in the upper left of the grid. Which will happen if the snake moves like this each step: step 1: left ... Options: 0: The snake hits the bound of the grid. 1: The snake hits its body. 2: The snake reaches the food. 3: Nothing happens.

# Answer: 1

GPT-4o Response: To determine what happens when the snake moves according to the given steps, let's track the position of the snake's head and body through each step: Initial position: - Head: (0, 4) - Body: (0, 3), (0, 2), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (1, 3), (1, 2) Step 1: Move left - Head moves to: (0, 3) New body: (0, 4), (0, 3) ... The snake hits the bound of the grid.

Ground Truth: At first the snake head(yellow block) is at $(0, 4)$ . The snake body(blue blocks) is at $[(1, 4) \ldots (2, 1)]$ . Then it moves like this: step 1: move left Before moving, the snake is at $[(0, 4) \ldots (2, 1)]$ . After moving left, the snake head will move left directly, which will be at $(0, 3)$ . Each block of the snake body will move to the position of the block in front of it, so the snake body will be at $[(0, 4) \ldots (3, 1)]$ . Now the snake hits its body.

Error Analysis: GPT-4o does not correctly identify all cells of the snake body, and does not realize the snake hits its body, though it states after step 1 head moves to (0, 3) and new body includes (0, 3).

![](images/fdbaf42453c9d0a32dff0d574ddf2b51c6ffed7e88692393212d3419b308b877.jpg)  
Figure 21 GPT-4o Snake case study

Question: On an 3x3 board, there are 3 levels (0 to 2). At each level k, the x and y coordinates range from 0 to 2-k, resulting in (2-k)***2 slots per level. Two players are designated as PLAYER_0 and PLAYER_1, with blue balls for PLAYER_0 and red balls for PLAYER_1 ... What is the status of the ball on Level 1, which has coordinate ([1, 1])? Options: 1. PLAYER_0 2. PLAYER_1 3. Empty 4. Index out of bound Answer: 2

GPT-4o Response: The image shows that there is no ball placed at coordinate $(1, 1)$ on Level 1. Therefore, the correct answer is: 4. Index out of bound

Ground Truth: Based on level 1's grid (specifically at coordinate [1, 1]), the ball is red, which corresponds to PLAYER_1.

Error Analysis: (1, 1) on Level 1 is marked as red, but GPT-40 reads it as empty, and wrongly thinks that empty fits option 4.

Figure 22 GPT-4o Pyramid Chess case study

![](images/d61c2589616a98f6067e8f5056e22162dc68265013f44d095955eb7a5760ce3c.jpg)  
Figure 23 GPT-4o Maze case study

Question: Which directions are available to move now? Options: A. up B. left C. right D. up, down E. left, right F. up, left, right G. down, left, right H. up, down, left, right

# Answer: C

GPT-4o Response: Up (0, 1): This is a blue block, so the player cannot move up. Down (2, 1): This is a white block, so the player can move down. Left (1, 0): This is a blue block, so the player cannot move left. Right (1, 2): This is a white block, so the player can move right. Therefore, the available directions for the player to move are down and right. The correct option is: D. up, down

# Ground Truth:

The player is on $(1, 1)$ , and for cells adjacent, only $(1, 2)$ is empty. The player can move right. Therefore, The option is C.

Error Analysis: GPT-4o incorrectly recognizes (2, 1) as white.

![](images/3a55a05d668874eab41cae2b3e5948f4beaa0e6c72b1f13d7417b1997d444c38.jpg)

![](images/54b8086223180ff4e4aa4805745844baf9045e14902e85014f871ea1f0a32e6f.jpg)

![](images/34f09e9c6424c5d2bcbbfd0fd514dd45d39f39fa8314b46fa71765ee6141d674.jpg)

![](images/d831b069ffacbdd1e5add91fb064612ca310862cb941cc09ebe375d7e2126273.jpg)

![](images/2658487f9992cd697aae96007ac8fc71de810141f5515b32d818da0ea07f5313.jpg)

![](images/5736eca98db9dba31b192edfe446bd217ef1d545c39bc6153cacc569acd71d71.jpg)

![](images/7da7a1a600332dd05538970558d534eca6a3f65af54c056e9a50b28e884528ad.jpg)

![](images/fb320155d23d2c59f631c3e4d3d7a76f182682223251efd2eea152c28cb90008.jpg)

![](images/a17ef2fd621288c60fb8f1c4077f667d23e816684fced315fb3a85a358580cf1.jpg)  
Figure 24 GPT-4o Ultra Tic-Tac-Toe case study

Question: In this game, the board is divided into 9 3*3 squares (called Nine-grids). We use (i, j, row, col) to represent the coordinates of a cell: (i, j) represents the coordinates of the Nine-grid; (row, col) represents the coordinate of the cell within the Nine-grid; i, j, row, col all range from 1 to 3. Two players take turns placing pieces on the board to mark the cells on the board, with the first player using "X" and the second player using "O" (this is the same as traditional TicTacToe). After the game starts, the first player places a chess piece in any cell in the Nine-grid in the middle (i.e., the Nine-grid (2, 2)). How many middle cells in the image are marked? Options: 1. 0 2. 1 3. 2 4. 3 5. 4 6. 5 7. 6 8. 7 9. 8 10. 9

Answer: 3

GPT-4o Response:

Top-left grid: (2,2) is unmarked. Top-middle grid: (2,2) is unmarked. Top-right grid: (2,2) is $\mathbf{X}$ (marked). Middle-left grid: (2,2) is $\mathbf{X}$ (marked). Center grid: (2,2) is unmarked. Middle-right grid: (2,2) is unmarked. Bottom-left grid: (2,2) is $\mathbf{O}$ (marked). Bottom-middle grid: (2,2) is unmarked. Bottom-right grid: (2,2) is $\mathbf{X}$ (marked).

Correct answer: 5.4

Ground Truth: By analyzing the blocks in the picture, we can see that the middle cells of nine-grid (1, 2) (Top-middle) and (3, 1) (Bottom-left) are marked, while the rest are empty. So, there are 2 middle cells marked, which means the answer is 3.

Error Analysis: GPT-40 misjudges the middle cells of Top-middle, Top-right, Middle-left, Bottom-right, and Bottom-right.

Typically, each game provides example images for three Plot Levels (Easy, Medium, Hard) representing different image complexities, along with their grading criteria. For demonstration purposes here, the images have been uniformly scaled. Please refer to the dataset repository for the actual relative sizes and resolutions of the images. The average height and width of the images in our dataset have been presented in Section 3.

For the 10 games:

1. The specific questions use these images as visual input.   
2. Labels such as "E1", "M2", and "H1" are used to denote specific images. For example, "Q1 (E1)" indicates that the corresponding image for this Q1 question sample is the "E1" image (i.e., the first image of the "Easy" Plot Level).   
3. For each game, its Introduction text is a common component presupposed to the beginning of every associated question.   
4. Due to space limitations, we have reasonably simplified the Introduction for some games, and we have also omitted some content from parts of the analyses. However, most analysis processes remain detailed and almost all clearly demonstrate the line of reasoning.

# J.1 3D Spatial Perception and Understanding

# J.1.1 3D Reconstruction

The game takes place in a 3x3x3 three-dimensional space with randomly initialized small cubes (voxels). Players reference two target side views (projections) and continue placing voxels in the 3D space to make the structure match these views (not considered in some tasks), with a maximum limit on placed voxels, all of which must be connected (not placed in midair). Question types include counting voxels in the current structure, identifying if given coordinates contain a voxel, checking if the current structure matches the side views, predicting side views after voxel additions, selecting the addition sequence that results in the structure matching side views, and calculating the minimum voxels needed to be added from the current structure to meet the two side views. The difficulty (Plot Level) is primarily determined by the number of voxels in the target three-dimensional structure.

<table><tr><td colspan="2">Easy
Final voxel count ∈ [3, 5]</td><td colspan="2">Medium
Final voxel count ∈ [6, 10]</td><td colspan="2">Hard
Final voxel count ∈ [11, 15]</td><td></td></tr><tr><td colspan="2">3D Voxel Reconstruction Game</td><td colspan="2">3D Voxel Reconstruction Game</td><td colspan="2">3D Voxel Reconstruction Game</td><td></td></tr><tr><td>Custom Structure</td><td>Front View (Y-Z Plane)</td><td>Current Structure</td><td>Front View (Y-Z plane)</td><td>Complete Solution</td><td>Front View (Y-Z plane)</td><td></td></tr><tr><td>Remaining Available Voxels: 0</td><td>1 2 3</td><td>1 2 3</td><td>1 2 3</td><td>1 2 3</td><td>1 2 3</td><td></td></tr><tr><td>E1</td><td>M1</td><td>H1</td><td colspan="2">3D Voxel Reconstruction Game</td><td colspan="2">3D Voxel Reconstruction Game</td></tr><tr><td colspan="2">3D Voxel Reconstruction Game</td><td colspan="2">3D Voxel Reconstruction Game</td><td colspan="2">3D Voxel Reconstruction Game</td><td></td></tr><tr><td>Current Structure</td><td>Front View (Y-Z plane)</td><td>Current Structure</td><td>Front View (Y-Z plane)</td><td>Current Structure</td><td>Front View (Y-Z plane)</td><td></td></tr><tr><td>Remaining Available Voxels: 1</td><td>1 2 3</td><td>1 2 3</td><td>1 2 3</td><td>1 2 3</td><td>1 2 3</td><td></td></tr><tr><td>E2</td><td>M2</td><td>H2</td><td colspan="2">3D Voxel Reconstruction Game</td><td colspan="2">3D Voxel Reconstruction Game</td></tr></table>

Question information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Count voxels in the 3D structure.</td></tr><tr><td>Q2</td><td>Target Perception</td><td>Easy</td><td>From options, select the position containing a voxel.</td></tr><tr><td>Q3</td><td>Target Perception</td><td>Medium</td><td>Select the option describing how the structure&#x27;s projections match target projections.</td></tr><tr><td>Q4</td><td>State Prediction</td><td>Medium</td><td>Predict projections after adding specified voxels.</td></tr><tr><td>Q5</td><td>State Prediction</td><td>Hard</td><td>Choose the correct voxel addition sequence to match target projection(s), adhering to game rules.</td></tr><tr><td>Q5</td><td>Strategy Optimization</td><td>Hard</td><td>Calculate the minimum additional voxels required to match both target projections.</td></tr></table>

# Specific questions and analysis

Introduction: The current structure has some initial voxels, and your goal is to complete it. Game Rules: 1. Goal: Reconstruct a 3D structure by adding voxels to match given projections.

2. Grid Space: The game is played on a $3 \times 3 \times 3$ cube grid.   
3. Coordinates: Position $(\mathrm{x},\mathrm{y},\mathrm{z})$ ranges from 1 to 3, with $(1,1,1)$ at front-left-bottom.   
4. Position Rule: Each position can contain at most one voxel.   
5. Connectivity: All voxels must be connected face-to-face.   
6. Voxel Limit: You have a maximum of n additional voxels available.

7. Placement Rule: New voxels can only be placed adjacent to existing ones.   
8. Front View (Y-Z): Shows structure when viewed along the negative X-axis direction (front to back), with Y as horizontal axis and Z as vertical axis. Projection coordinates are in $(y,z)$ format.   
9. Side View (X-Z): Shows structure when viewed along the positive Y-axis direction (left to right), with X as horizontal axis and Z as vertical axis. Projection coordinates are in $(\mathrm{x},\mathrm{z})$ format.

Q1 (E1): How many voxels are there in the given structure?

Analysis: The structure contains voxels at the following positions: $(2,1,1)$ , $(2,2,1)$ . By counting these positions, we can see there are 2 voxels in total. Therefore the answer is 2.

Q2 (M1): Which of the following positions contains a voxel? Choose the correct position from the options below.

Options: 1: (3,2,2); 2: (3,2,1); 3: (2,3,1); 4: (2,2,3); 5: (2,1,1); 6: (1,3,3)

Analysis: Let's analyze each option:

Option 1 - Position (3,2,2): This position is empty. ... (omitted)

Option 5 - Position (2,1,1): This position contains a voxel. This is the correct answer.

Option 6 - Position (1,3,3): This position is empty.

Therefore, the correct answer is option 5.

Q3 (H1): How does the voxel structure's projections match with the target projections?

Choose the correct description from the options below.

Options:

1: Neither Y-Z projection nor X-Z projection matches the target;   
2: Only Y-Z projection matches the target; 3: Only X-Z projection matches the target;   
4: Both Y-Z and X-Z projections match the target

Analysis: Let's analyze the projections:

1. Looking along the negative X-axis direction (Front View, using (y,z) coordinates): - We can see voxels at positions [(2, 1, 2), ... (omitted), (3, 3, 3)], forming a Y-Z projection of [(1, 1), ... (omitted), (3, 3)] - This matches the target Y-Z projection exactly.   
2. Looking along the positive Y-axis direction (Side View, using (x,z) coordinates): - We can see voxels at positions [(1, 1, 1), ... (omitted), (3, 3, 2)], forming a X-Z projection of [(1, 1), ... (omitted), (3, 3)] - This matches the target X-Z projection exactly.

Based on the above analysis, both projections match the target. Therefore, the correct answer is option 4.

Q4 (E2): Action: Add 1 voxels at positions: [(2, 2, 1)]

Question: After adding these voxels, what will be the X-Z projection of the new structure? Answer Format:

1. Write the answer as a list of three lists: [[row1], [row2], [row3]] 2. Each row should contain three numbers (0 or 1) 3. Rows are ordered from top to bottom of the projection 4. Numbers in each row are ordered from left to right 5. Use 1 to indicate presence of a voxel in the projection, 0 for empty space 6. Example format: [[0, 1, 0], [1, 1, 0], [0, 1, 1]]

Analysis: Let's analyze the projection:

Looking along the positive Y-axis direction (Side View, using (x,z) coordinates):

- We can see voxels at positions [(2, 2, 1)], which in X-Z projection appear at positions [(2, 1)].

Therefore, the answer is: $[ [0, 0, 0 ], [0, 0, 0 ], [0, 1, 0 ]]$

Q5 (M2): Which sequence of voxel additions will make the structure match the both target projections?

Choose the correct sequence from the options below.

Options:

1: Add voxels at positions: [(1, 1, 1), (1, 1, 2), (1, 2, 1), (3, 2, 1)]; ... (omitted)   
6: Add voxels at positions: [(1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 2, 3)]; ... (omitted)   
8: Add voxels at positions: [(2, 1, 2), (2, 3, 1), (3, 3, 3)]

Analysis: Let's analyze each option:

Current structure: $[(2,2,1),(2,2,2)]$

Option 1: - The added voxels maintain connectivity - Does not match both target projections -

Uses 4 voxels, which is within the limit of 4

... (omitted)

Option 6: - The added voxels maintain connectivity - Matches both target projections - Uses 4 voxels, which is within the limit of 4

... (omitted)

Option 8: - The added voxels are not all connected to the existing structure - Does not match

both target projections - Uses 3 voxels, which is within the limit of 4

Therefore, the correct answer is option 6.

Q6 (H2): What is the minimum number of voxels needed to add to the current structure to make it match both target projections?

Analysis: Let's solve this optimization problem through systematic reasoning:

1. Basic Information: - Current structure: 6 voxels at positions [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1,

2), (3, 1, 1), (3, 2, 1)] - Remaining available voxels: 3

2. Analysis of Y-Z Projection (Front View):

a) Current Y-Z projection: $[0, 0, 0]$ (top) $[1, 0, 0]$ (middle) $[1, 1, 0]$ (bottom)

b) Target Y-Z projection: [1, 1, 0] (top) [1, 1, 0] (middle) [1, 1, 0] (bottom)

c) Candidate positions from Y-Z view: $(?, 1, 3)$ , $(?, 2, 2)$ , $(?, 2, 3)$ where ? can be any value from 1 to 3 for x-coordinate

d) Note: At positions where projection already shows 1, we can add more voxels without affecting the projection. For example, if (2, y0, z0) exists (where y0 and z0 are specific fixed values), we can add (1, y0, z0) or (3, y0, z0) at the same projection position.

3. Analysis of X-Z Projection (Side View): ... (omitted)

4. Finding Required Positions:

By matching candidates from both projections:

- When $(?,\mathrm{y},\mathrm{z})$ from Y-Z view matches (x, ?, z) from X-Z view, position (x, y, z) can be filled.

- Example: if we have $(?, 2, 3)$ and $(1,?, 3)$ , then $(1, 2, 3)$ is required

- To ensure connectivity, we can add voxels at positions where projections already show 1

* This strategy is optimal because it doesn't create new projections

* Use these positions as 'bridges' to connect required positions Required positions from projection matching: [(1, 1, 3), (2, 2, 2), (2, 2, 3)]

5. Connectivity Analysis and Completion: ... (omitted)

6. Verifying Optimality: ... (omitted)

Therefore, the minimum number of voxels needed to complete the reconstruction is 3.

# J.1.2 3D Maze

This game involves pathfinding within a three-dimensional maze constructed from unit cubes arranged in a 3D grid space (voxel-based). Traversal is subject to specific rules: horizontal movement (along X or Y axes) is permitted between adjacent cubes only if they reside at the same height (Z-coordinate). Vertical movement (ascending/descending along the Z-axis) is permitted between vertically aligned cubes (sharing X and Y coordinates) only if a ladder explicitly connects them. Key locations are color-coded: a blue cube designates the starting position, and a red cube marks the goal destination. Additionally, green cubes, often labeled with numbers, serve as waypoints, decision junctions, or specific points of interest referenced in the questions. Question types assess spatial navigation and path analysis: (1) determining the correct direction of

travel required at each green waypoint to follow a path towards the destination; (2) ordering a set of specified green cubes based on their height (Z-coordinate); (3) identifying the sequence of green cubes visited along the shortest path from the start to the end; (4) reporting the exact sequence in which green cubes are encountered when traversing from start to end following a defined path. Path generation often utilizes concatenation of randomized 'atomic' path segments (e.g., move $+2\mathrm{X}$ , move $+2\mathrm{Y}$ , move $+2/3\mathrm{Z}$ ) to create a primary route, with branching paths potentially added similarly to introduce choices, aiming to minimize visual occlusion between path segments.

# Images and Plot Level division

<table><tr><td>Easy
Simple, a single path</td><td>Medium
Complex, side road exists</td></tr><tr><td>E1</td><td>M1</td></tr><tr><td>E2</td><td>M2</td></tr></table>

Question information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Height Comparison</td></tr><tr><td>Q2</td><td>State Prediction</td><td>Medium</td><td>Sequence Finding</td></tr><tr><td>Q3</td><td>State Prediction</td><td>Medium</td><td>Main Path</td></tr><tr><td>Q4</td><td>State Prediction</td><td>Hard</td><td>Path Finding</td></tr></table>

# Specific questions and analysis

Introduction: Rules: 1. Player can only walk on top of cubes

2. Player can climb ladders if they can reach the cube under the ladder   
3. From a ladder, player can reach the top of the last cube with the ladder   
4. Blue cube is start position, red cube is goal position

5. Green cubes are numbered points (1, 2, and 3)

Q1 (E1): What is the correct height relationship between the three numbered points? Use $<$ for 'lower than' and $=$ for 'same height as'.

Options:

$$
\begin{array}{l} 1: 2 = 3 <   1 2: 1 <   3 <   2 3: 3 <   1 <   2 4: 1 <   2 = 3 \\ 5: 3 <   2 <   1 6: 2 <   1 = 3 7: 1 = 2 = 3 8: 3 <   1 = 2 \\ \end{array}
$$

Analysis: Analyzing the heights of each point:

Comparing points 1 and 2: Found a path from 1 to 2:   
* Move left-forward * Move left-forward   
- Point 2 is same height as point 1   
Comparing points 1 and 3: Found a path from 3 to 1:   
* Go up 3 blocks * Go up 3 blocks   
- Point 3 is lower than point 1   
Comparing points 2 and 3: Found a path from 3 to 2:   
* Go up 3 blocks * Go up 3 blocks * Move left-forward * Move left-forward   
- Point 3 is lower than point 2   
Therefore, the correct height relationship is $3 < 1 = 2$ , making the answer Option 8.

Q2 (E2): What is the correct sequence of numbered checkpoints when following the path from start to goal? Options:

1: Start -> 2 -> 3 -> 1 -> 4 -> Goal; 2: Start -> 2 -> 3 -> 4 -> 1 -> Goal;   
3: Start -> 4 -> 3 -> 1 -> 2 -> Goal; 4: Start -> 4 -> 2 -> 3 -> 1 -> Goal;   
5: Start -> 3 -> 2 -> 4 -> 1 -> Goal; 6: Start -> 2 -> 4 -> 3 -> 1 -> Goal

Analysis: Following the path from start to goal:

- Step 1: Move right-forward - Step 2: At checkpoint 2 - Step 3: Move up   
- Step 4: Move left-forward 
- Step 5: At checkpoint 3 
- Step 6: Move left-forward   
- Step 7: Move up - Step 8: At checkpoint 4 - Step 9: Move right-forward   
- Step 10: At checkpoint 1 - Step 11: Move right-forward

Therefore, the correct sequence is Start $- > 2 - > 3 - > 4 - > 1 - >$ Goal, making the answer Option 2.

Q3 (M1): Which numbered blocks are passed through when following the most direct path from start to goal?

Options:

1: 1, 2; 2: 2, 3; 3: 3; 4: 2; 5: 1; 6: None; 7: 1, 2, 3; 8: 1, 3

Analysis: Following the main path from start to goal:

- Step 1: Move up - Step 3: Move up - Step 4: Move right-forward   
- Step 5: Move left-forward - Step 6: Move right-forward - Step 7: Move right-forward

Blocks not on main path: 1, 2. Therefore, the blocks passed through on the main path are: 3,

making the answer Option 3.

Q4 (M2): Which combination of path choices leads to the goal?

Options:

1: 1-right-forward, 2-right-forward, 3-up;   
2: 1-left-forward, 2-right-forward, 3-left-forward;   
3: 1-left-forward, 2-up, 3-left-forward; 4: 1-left-forward, 2-up, 3-up;   
5: 1-left-forward, 2-right-forward, 3-up; 6: 1-right-forward, 2-up, 3-up;   
7: 1-right-forward, 2-right-forward, 3-left-forward;   
8: 1-right-forward, 2-up, 3-left-forward

Analysis: From the start point, you first meet branch 1, then branch 2, then branch 3, before finally reaching the goal.

Analyzing each branch point:

- At branch 1, going right-forward leads to branch 2, while going left-forward leads to a dead end   
- At branch 2, going up leads to branch 3, while going right-forward leads to a dead end   
- At branch 3, going up leads toward the goal, while going left-forward leads to a dead end

Therefore, the correct sequence is 1-right-forward, 2-up, 3-up, that is 1-right-forward, 2-up, 3-up, making the answer Option 6.

# J.1.3 Rubik's Cube

This game is based on the classic Rubik's Cube puzzle. The game interface presents both 3D views and an unfolded view of the cube. The 3D views display the cube from two different angles: left-tilted 30 degrees looking down, and right-tilted 30 degrees looking up. The cube features six faces with distinct colors (yellow, white, orange, red, blue, and green), and players can manipulate the cube according to standard rotation rules (where F, B, L, R, U, D represent Front, Back, Left, Right, Upper, and Down faces, with a prime symbol denoting counterclockwise rotation).

Question types, assessing spatial reasoning and pattern recognition, include identifying the color at a specific position on a face, counting a color's occurrences on a face, and predicting a position's color after a move sequence. Further questions ask for the minimum moves to solve a single face or the entire cube. The difficulty level (Plot Level) is determined by the number of random moves used to scramble the cube: 1 move for Easy, 2 moves for Medium, and 3 moves for Hard.

# Images and Plot Level division

<table><tr><td colspan="2">Easy
1 random move</td><td colspan="2">Medium
2 random moves</td><td colspan="2">Hard
3 random moves</td></tr><tr><td>D
L
R
B
F
U
Coordinate Reference
(2,0), (2,1), (2,2)
(1,0), (1,1), (1,2)
(0,0), (0,1), (0,2)</td><td>Front view (F, R, U faces)
F
R
U
Back view (L, D, B faces)
L
D
B
Coordinate Reference
(2,0), (2,1), (2,2)
(1,0), (1,1), (1,2)
(0,0), (0,1), (0,2)</td><td>D
L
R
B
F
U
Coordinate Reference
(2,0), (2,1), (2,2)
(1,0), (1,1), (1,2)
(0,0), (0,1), (0,2)</td><td>Front view (F, R, U faces)
F
R
U
Back view (L, D, B faces)
L
D
B
Coordinate Reference
(2.0), (2.1), (2.2)
(1.0), (1.1), (1.2)
(0.0), (0.1), (0.2)</td><td>D
L
R
B
F
U
Coordinate Reference
(2,0), (2,1), (2,2)
(1,0), (1.1), (1.2)
(0.0), (0.1), (0.2)</td><td>Front view (F, R, U faces)
F
R
U
Back view (L, D, B faces)
L
D
B
Coordinate Reference
(2,0), (2,1), (2,2)
(1,0), (1.1), (1.2)
(0.0), (0.1), (0.2)</td></tr></table>

# J.1.4 Pyramid Chess

This is a 3D two-player competitive game. Players take turns placing balls on a board, building a pyramid structure layer by layer. The player whose ball occupies the pyramid's top wins.

Question types challenge players to assess the board by determining which player's ball occupies a given position, the specific state of any board position, and the total count of balls. Additionally, questions involve predicting the result of a player placing a ball, calculating the minimum number of moves required to place a

ball at a certain position, and identifying a player's optimal placement in the current state. Plot Level is determined by the board's base size, with larger base dimensions increasing the challenge.

![](images/ece606cc2e69a8e8c2033555be5c167ac83711a6503ca918adab3a20b9f3894d.jpg)  
Easy   
Level $03\times 3$   
Medium   
Level $04\times 4$   
Hard   
Level $0.5 \times 5$

![](images/006deb782f2d2b5eb8db0224f1376d873c6995759312ce1c9253c5f8c499cbb1.jpg)

![](images/834283e382cbdf4286897964d641a9956e343266b93bee9bc7777944eff15752.jpg)

# J.1.5 Minecraft

This Minecraft QA generator is designed to produce a series of questions that test 3D perception and understanding within a simulated "Minecraft" environment. Given the open-ended nature of Minecraft, the tasks are custom-designed to probe specific cognitive abilities. The generated questions aim to evaluate how well an agent can interpret and reason about 3D scenes.

The question set begins by assessing precise 3D perception. Q1 requires recognizing various scenarios present in the image, such as different ores, TNT, pumpkins, or environmental features like rivers and lava. Q2 tests the ability to accurately count the total number of blocks in a given structure. These foundational perceptual skills are prerequisites for the subsequent three tasks, which demand reasoning based on both visual input and provided rules. These more complex questions involve planning: determining the minimum blocks to cross a river (Q3), calculating the blocks needed to reach a target block at a certain height, possibly using ladders (Q4), and a combined scenario requiring both river crossing and climbing to access a target block, again considering ladders (Q5). Plot Level is determined by the number of scenarios (Q1), the cuboid size (Q2), the width of the river (Q3, Q5) and the height of the target block (Q4, Q5).

![](images/09c5cbe2c9fd02c0497ae151b402c2e30349e839bd9a1cec7158f526094ee6d6.jpg)  
For Q1

![](images/acea6c4d8e99f81f72c5eb5d30ce5fb26dc4cf0c1c94af7276277dfe9d6a38f8.jpg)  
For Q3

![](images/800e433de7ca97be2005bd2dc9b6ac650c329ffbdc3ef0fae89b012db5f4afff.jpg)  
For Q4

![](images/4d3317e61210e6dc0ca77abeb76e8782e056501acc2f7b53eac2e78c7564a817.jpg)  
For Q2

![](images/2cc0f797a136a7c68fde9357f5d63b389a870d67a01bb4b54ef2511e665832cf.jpg)  
For Q4

![](images/eaa27cb1c81afc7f9f3dee9a4545ceb42000f64f49631957d7e9dee4af2abfb9.jpg)  
For Q5

# J.2 Pattern Recognition and Matching

# J.2.1 Color Hue

This game involves reasoning about color gradients within a grid structure. Certain rows and/or columns within the grid display smooth color transitions. Cells that are intentionally left blank or empty are visually marked with a gray crosshatch pattern. Color information may be conveyed using standard color names (e.g., "purple"), derived programmatically from their Hue-Saturation-Value (HSV) properties.

Question types focus on understanding and interpolating these color gradients: (1) identifying the specific color present at a given row and column index; (2) determining the starting and ending colors of a specified gradient row or column; (3) selecting the correct color from a provided set of options (e.g., six color patches) that should logically fill a designated blank cell (marked with a letter) based on the surrounding gradient(s). The complexity ('plot level') scales with the dimensions of the grid.

# Images and Plot Level division

![](images/ac160124530d170f2ef01b8f64c0599bf789b1948fbb7c4194e5c6ce9f1a50e6.jpg)  
Easy   
$5\times 5$   
E

![](images/85d1e0dae3b01d51c13a8f643fc98f61acfa871df91aef857501745931fa7ff4.jpg)  
Medium   
$6\times 6$   
M

![](images/77d4b290a9d03ff8257b28f407db8847cb615929d79e99881d03678f5da338df.jpg)  
Hard   
$8\times 8$   
H

# Question information

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Color Description</td></tr><tr><td>Q2</td><td>Target Perception</td><td>Medium</td><td>Gradient Pattern</td></tr><tr><td>Q3</td><td>State Prediction</td><td>Hard</td><td>Color Matching</td></tr></table>

# Specific questions and analysis

# Introduction: Rules:

1. Each numbered region represents a piece on the board.   
2. Pieces are considered adjacent if they share at least one edge.   
3. Pieces that only touch at corners are not considered adjacent.   
4. Some pieces have been removed and are shown below the main board.

Q1 (M): What color is the cell at row 1, column 6?

Options:

1: green; 2: white; 3: vivid red; 4: pale bright green; 5: bright cyan; 6: cyan; 7: bright orange; 8: dark green.

Analysis: The cell at position $(1,6)$ is pale bright green. So the answer is Option 4.

Q2 (E): What is the gradient pattern in column 5?

Options:   
1: transitioning from bright purple to pale dark cyan;   
2: transitioning from vivid green to pale bright purple;   
3: transitioning from pale bright yellow to vivid dark blue;   
4: transitioning from vivid bright blue to pale red;   
5: transitioning from yellow to light gray;   
6: transitioning from red to pale bright cyan;   
7: transitioning from purple to bright red;   
8: transitioning from black to pale bright cyan.

Analysis: The column 5 shows a pattern that is transitioning from purple to bright red. So the answer is Option 7.

Q3 (H): Which color should be put in cell B?

Options: Colors are numbered from 1 to 6 in the palette below.

Analysis: We need to find the correct color for cell B at position (2, 6). Let's analyze the color patterns around this cell:

Looking vertically, we see a pattern transitioning from pale bright red to bright yellow. Let's look at our color options:

Option 1 is bright yellow; Option 2 is vivid dark purple; Option 3 is pale yellow; Option 4 is vivid bright indigo; Option 5 is yellow; Option 6 is pale bright red. Based on the pattern, we should use pale bright red (Option 6).

# J.2.2 Tangram

This game presents a spatial reasoning puzzle inspired by Tangram, involving the manipulation and fitting of polygonal shapes within a grid. The grid is partitioned into several distinct regions or "pieces", each identified by a unique integer ID. Cells belonging to a specific piece display that piece's ID number; cells not part of any displayed piece are left blank, representing empty space. One or more pieces are removed from the main board. Questions test pattern recognition and spatial matching skills across various dimensions: identifying piece area and adjacency, determining correct rotations to fit removed pieces back into empty spaces, and strategically positioning multiple pieces to fill available gaps. The puzzle complexity scales with the grid size.

Images and Plot Level division

<table><tr><td colspan="3">Easy
5×5 Main Board</td><td colspan="101">Medium
8×8 Main Board</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2" colspan="3">Main Board</td><td colspan="99">Main Board</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td><td>24</td><td>25</td><td>26</td><td>27</td><td>28</td><td>29</td><td>30</td><td>31</td><td>32</td><td>33</td><td>34</td><td>35</td><td>36</td><td>37</td><td>38</td><td>39</td><td>40</td><td>41</td><td>42</td><td>43</td><td>44</td><td>45</td><td>46</td><td>47</td><td>48</td><td>49</td><td>50</td><td>51</td><td>52</td><td>53</td><td>54</td><td>55</td><td>56</td><td>57</td><td>58</td><td>59</td><td>60</td><td>61</td><td>62</td><td>63</td><td>64</td><td>65</td><td>66</td><td>67</td><td>68</td><td>69</td><td>70</td><td>71</td><td>72</td><td>73</td><td>74</td><td>75</td><td>76</td><td>77</td><td>78</td><td>79</td><td>80</td><td>81</td><td>82</td><td>83</td><td>84</td><td>85</td><td>86</td><td>87</td><td>88</td><td>89</td><td>90</td><td>91</td><td>92</td><td>93</td><td>94</td><td>95</td><td>96</td><td>97</td><td>98</td><td>99</td><td>100</td><td>101</td><td>102</td><td>103</td><td>104</td><td>105</td><td>106</td><td>107</td><td>108</td><td>109</td><td>110</td><td>111</td><td>112</td><td>113</td><td>114</td><td>115</td><td>116</td><td>117</td><td>118</td><td>119</td><td>120</td><td>121</td><td>122</td><td>123</td><td>124</td><td>125</td><td>126</td><td>127</td><td>128</td><td>129</td><td>130</td><td>131</td><td>132</td><td>133</td><td>134</td><td>135</td><td>136</td><td>137</td><td>138</td><td>139</td><td>140</td><td>141</td><td>142</td><td>143</td><td>144</td><td>145</td><td>146</td><td>147</td><td>148</td><td>149</td><td>150</td><td>151</td><td>152</td><td>153</td><td>154</td><td>155</td><td>156</td><td>157</td><td>158</td><td>159</td><td>160</td><td>161</td><td>162</td><td>163</td><td>164</td><td>165</td><td>166</td><td>167</td><td>168</td><td>169</td><td>170</td><td>171</td><td>172</td><td>173</td><td>174</td><td>175</td><td>176</td><td>177</td><td>178</td><td>179</td><td>180</td><td>181</td><td>182</td><td>183</td><td>184</td><td>185</td><td>186</td><td>187</td><td>188</td><td>189</td><td>190</td><td>191</td><td>192</td><td>193</td><td>194</td><td>195</td><td>196</td><td>197</td><td>198</td><td>199</td><td>200</td><td>201</td><td>202</td><td>203</td><td>204</td><td>205</td><td>206</td><td>207</td><td>208</td><td>209</td><td>210</td><td>211</td><td>212</td><td>213</td><td>214</td><td>215</td><td>216</td><td>217</td><td>218</td><td>219</td><td>220</td><td>221</td><td>222</td><td>223</td><td>224</td><td>225</td><td>226</td><td>227</td><td>228</td><td>229</td><td>230</td><td>231</td><td>232</td><td>233</td><td>234</td><td>235</td><td>236</td><td>237</td><td>238</td><td>239</td><td>240</td><td>241</td><td>242</td><td>243</td><td>244</td><td>245</td><td>246</td><td>247</td><td>248</td><td>249</td><td>250</td><td>251</td><td>252</td><td>253</td><td>254</td><td>255</td><td>256</td><td>257</td><td>258</td><td>259</td><td>260</td><td>261</td><td>262</td><td>263</td><td>264</td><td>265</td><td>266</td><td>267</td><td>268</td><td>269</td><td>270</td><td>271</td><td>272</td><td>273</td><td>274</td><td>275</td><td>276</td><td>277</td><td>278</td><td>279</td><td>280</td><td>281</td><td>282</td><td>283</td><td>284</td><td>285</td><td>286</td><td>287</td><td>288</td><td>289</td><td>290</td><td>291</td><td>292</td><td>293</td><td>294</td><td>295</td><td>296</td><td>297</td><td>298</td><td>299</td><td>300</td><td>301</td><td>302</td><td>303</td><td>304</td><td>305</td><td>306</td><td>307</td><td>308</td><td>309</td><td>310</td><td>311</td><td>312</td><td>313</td><td>314</td><td>315</td><td>316</td><td>317</td><td>318</td><td>319</td><td>320</td><td>321</td><td>322</td><td>323</td><td>324</td><td>325</td><td>326</td><td>327</td><td>328</td><td>329</td><td>330</td><td>331</td><td>332</td><td>333</td><td>334</td><td>335</td><td>336</td><td>337</td><td>338</td><td>339</td><td>340</td><td>341</td><td>342</td><td>343</td><td>344</td><td>345</td><td>346</td><td>347</td><td>348</td><td>349</td><td>350</td><td>351</td><td>352</td><td>353</td><td>354</td><td>355</td><td>356</td><td>357</td><td>358</td><td>359</td><td>360</td><td>361</td><td>362</td><td>363</td><td>364</td><td>365</td><td>366</td><td>367</td><td>368</td><td>369</td><td>370</td><td>371</td><td>372</td><td>373</td><td>374</td><td>375</td><td>376</td><td>377</td><td>378</td><td>379</td><td>380</td><td>381</td><td>382</td><td>383</td><td>384</td><td>385</td><td>386</td><td>387</td><td>388</td><td>389</td><td>390</td><td>391</td><td>392</td><td>393</td><td>394</td><td>395</td><td>396</td><td>397</td><td>398</td><td>399</td><td>400</td><td>401</td><td>402</td><td>403</td><td>404</td><td>405</td><td>406</td><td>407</td><td>408</td><td>409</td><td>410</td><td>411</td><td>412</td><td>413</td><td>414</td><td>415</td><td>416</td><td>417</td><td>418</td><td>419</td><td>420</td><td>421</td><td>422</td><td>423</td><td>424</td><td>425</td><td>426</td><td>427</td><td>428</td><td>429</td><td>430</td><td>431</td><td>432</td><td>433</td><td>434</td><td>435</td><td>436</td><td>437</td><td>438</td><td>439</td><td>440</td><td>441</td><td>442</td><td>443</td><td>444</td><td>445</td><td>446</td><td>447</td><td>448</td><td>449</td><td>450</td><td>451</td><td>452</td><td>453</td><td>454</td><td>455</td><td>456</td><td>457</td><td>458</td><td>459</td><td>460</td><td>461</td><td>462</td><td>463</td><td>464</td><td>465</td><td>466</td><td>467</td><td>468</td><td>469</td><td>H1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Question information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Main board piece</td></tr><tr><td>Q2</td><td>State Prediction</td><td>Medium</td><td>Removed piece rotation feasibility</td></tr><tr><td>Q3</td><td>Target Perception</td><td>Medium</td><td>Target piece area calculation</td></tr><tr><td>Q4</td><td>Target Perception</td><td>Medium</td><td>Adjacent piece type count</td></tr><tr><td>Q5</td><td>State Prediction</td><td>Hard</td><td>Piece Placement</td></tr></table>

# Specific questions and analysis

Introduction: Rules:

1. Each numbered region represents a piece on the board.   
2. Pieces are considered adjacent if they share at least one edge.   
3. Pieces that only touch at corners are not considered adjacent.   
4. Some pieces have been removed and are shown below the main board.

Q1 (E1): How many pieces are currently on the main board?

Options: 1: 4 2: 1 3: 0 4: 3 5: 2 6: 5 7: 7 8: 6

Analysis: Let's analyze the puzzle state:

Pieces currently on the board: Piece 2 (vivid bright orange) around position (3, 3)

Removed pieces: Piece 4; Piece 3; Piece 1

By counting the unique non-zero numbers on the main board, we can see there are 1 pieces remaining. Therefore, the answer is Option 2.

Q2 (M1): Can the removed piece fit back into the main board by only rotation? If yes, what rotation(s) would work?

Options: 1: rotate 0 degrees 2: no matter what degrees rotated, it always can fit 3: both rotate 0 and 180 degrees 4: rotate 90 degrees clockwise 5: can't put inside (flipped) 6: rotate 180 degrees 7: rotate 90 degrees by both direction 8: rotate 90 degrees counterclockwise

Analysis: Let's analyze how piece 1 can be rotated to fit the hole:

1. Dimension Analysis: - Hole dimensions: 6x6 - Piece dimensions: 6x6

- Based on dimensions, these rotations (clockwise) might work: 0, 90, 180, 270

2. Testing Each Rotation:

$0^{\circ}$ rotation: - Failed: First mismatch at row 0, column 1 (mapped to $(0,1)$ of removed piece and

(2,3) of board) - At this position, the hole was empty but the piece was absent

... (omitted: $90^{\circ}$ and $180^{\circ}$ rotation)

$270^{\circ}$ rotation: - Success! Piece fits perfectly

3. Summary: - Valid rotations found: 270 - Some rotations work

So, the correct answer is: rotate 90 degrees counterclockwise which is Option 8.

Q3 (M2): Question: What is the area (number of cells) of Piece 1?

Options: 1: 11 2: 12 3: 14 4: 17 5: 18 6: 19 7: 20 8: 21

Analysis: Let's analyze Piece 1 (vivid blue) row by row:

The piece spans from row 0 to 5 (height of 6):

Row 0: 2 cells from column 0 to 1; ... (omitted)

Adding up all the cells: $2 + 2 + 3 + 4 + 4 + 2 = 17$ cells. Therefore, the answer is Option 4.

Q4 (H1): Question: How many different pieces are adjacent to Piece 3?

Options: 1: 5 2: 6 3: 0 4: 2 5: 3 6: 4 7: 7 8: 1

Analysis: Let's analyze Piece 3 (green):

Piece Boundaries: - Spans rows 3 to 7 (height: 5) - Spans columns 4 to 8 (width: 5)

1. Cell-by-cell examination: Cell (3,7): No adjacent pieces; ... (omitted) Cell (5,4): - down: Piece 4 (vivid bright red) at (6,4) ... (omitted) Cell (7,8): - down: Piece 4 (vivid bright red) at (8,8)

2. Adjacent Pieces Summary: - Piece 4 (vivid bright red): 7 contact sides

Total number of unique adjacent pieces: 1. Therefore, the answer is Option 8.

Q5 (H2): Question: At which position should Piece 1 be placed? Each option shows (top_row,left_col) to (bottom_row,right_col).

Options: 1: (0,3) to (6,6) 2: (0,6) to (6,9) 3: (0,4) to (6,7) 4: (0,5) to (6,8)

Analysis: Let's analyze the placement of Piece 1 and Piece 4:

1. Hole dimensions: 7x5 2. Piece 1 dimensions: 7x4 3. Piece 4 dimensions: 4x4

We know that if Piece 1 fits, then it must be placed at one of the four corners.

Testing each corner: - upper-left: Attempting to place Piece 1 at (0,5) to (6,8) Failed: Cell (4,0) on Removed Pieces plot maps to board position (4,5) which isn't empty - upper-right:

Attempting to place Piece 1 at (0,6) to (6,9) Success! Remaining hole dimensions: 4x4 Then placing Piece 4 at (0,5) to (3,8) Both pieces fit perfectly! - bottom-left: Since Piece 1 has the same height as the hole, bottom-left corner is same as upper-left corner. Skipped. ... (omitted: bottom-right, same as upper-right corner)

Therefore, Piece 1 should be placed at position (0,6) to (6,9) as shown in Option 2.

# J.2.3 Freecell

This scene presents a solitaire card game whose goal is to move all cards to foundation piles, following specific rules. This game is played with a standard deck of 52 cards, arranged in n tableau columns, four open cells, and four foundation piles. Cards can be moved between tableau columns according to descending order and alternating colors, while empty tableau spaces can only be filled by kings. The four open cells act as temporary storage, allowing players to temporarily hold cards for strategic moves. Complexity is controlled by adjusting the number of tableau columns, testing the model's ability to search for an efficient operation list to complete the game.

![](images/e9564673da5608f99c48405ad0003166b5232dd5e6ce4b6c8152c5f67221cd6a.jpg)

# J.2.4 Tetris

This Tetris-derived game maintains the original objectives while simplifying visuals to highlight core information. Players arrange falling blocks to eliminate rows by: moving/rotating pieces during descent until they land at the bottom or on other blocks, clearing complete horizontal rows. The game ends when blocks reach the grid's top. The simplified interface shows a white grid with gray squares representing placed blocks and red squares indicating the current falling piece (with grid coordinates). While actual games use color-coding for different block batches, this visual distinction is omitted as irrelevant to gameplay logic. Advanced Tetris variants are excluded.

Questions cover: 1) Empty squares in a specified row 2) Identifying the current red block's shape 3) Timesteps until the falling block lands after given moves 4) Maximum eliminable rows from the current block's optimal placement.

![](images/4e420b8a84a796ebcd0c4a3f411f7a1265bf85bde93c4588e1cda8d1a40fcbee.jpg)

# J.2.5 Zuma

This game is a classic marble-shooting puzzle game where players control a frog that shoots colored marbles toward a chain of rolling marbles on a track. The objective is to clear all marbles before they reach the black

hole at the end. Players must create groups of three or more same-colored marbles, which will disappear from the track. The frog's marbles travel in a straight line until they hit marbles already in their path.

The game tests spatial reasoning, color recognition, and strategic planning through various question types: identifying the color of the next marble to be shot, counting marbles of specific colors, determining the number of same-colored marble groups in certain directions, predicting which marble will be hit at specific angles, analyzing the outcome of shots, and evaluating optimal elimination strategies. Plot difficulty levels are determined by track length and marble count.

![](images/8e50437413470e5fc77a78940f1b0c647a0241872e49f2fb5688c0ed5a1e8276.jpg)

# J.2.6 Spider Solitaire

The game is based on Microsoft's classic Spider Solitaire, with the original four suits simplified to just one suit. The objective of the game is to move all 13 cards of the same suit, arranged in descending order from King to Ace, from the waste piles to the foundation piles. The cards in the waste piles must be arranged in descending order. The foundation piles serve as the final destination for complete sequences. The game screen includes several waste piles, a stock pile, and foundation piles, with each pile containing several stacked cards. Some cards are face down, indicating that their rank is unknown, while others are face up, revealing their rank. The dataset includes tasks such as identifying the card on top of a pile, moving cards from the waste piles, and determining the optimal move. The dataset is divided into three difficulty levels based on the number of waste piles.

![](images/f0c1d78d7af6df1d79d307818a9e38172d8f028e3c0d4613806820e060559ac4.jpg)

# J.2.7 Jewel2

Jewel2 is a grid-based strategic puzzle game. It is inspired by Microsoft's classic game Bejeweled 2, with certain modifications made to the original game. The game board is square-shaped and consists of five basic elements and seven special elements. The basic elements are five gems of the same shape but different colors, while the special elements are seven gems with different shapes from the basic ones, designed to test the model's pattern recognition ability. The main objective of the game is to eliminate elements by forming horizontal or vertical lines of three or more identical items. Successfully eliminating elements increases your

score and clears space for new elements to appear. The game tasks include recognizing elements on the board, executing elimination operations, and maximizing the score. Plot Level is determined by the size of the board

![](images/0a9a2766b93d72c2190072a2e98535405c45f4b40e63f8e4030ba2a1ad626971.jpg)  
Easy   
$4\times 4$   
Medium   
$5\times 5$   
Hard   
$6\times 6$   
Jewel2 Game   
Total Cleared: 44

![](images/07a5ac0520fe7fded88a4ba46f59896756c102a0910648b270cf234d642a2829.jpg)  
Jewel2 Game   
Total Cleared: 95

![](images/961871f37037052e2d95068704ffbf9c34ba173d6e51d9b34ff122738a96eed8.jpg)  
Jewel2 Game   
Total Cleared: 1

# J.2.8 Klondike

This Klondike Solitaire-based strategy game challenges players to analyze card layouts and apply rules for optimal decisions. It uses a standard interface with Stock, Waste, Foundation, and Tableau piles. The goal is to move all 52 cards, by suit and in ascending order (Ace to King), to the four Foundation Piles. Key mechanics include building Tableau piles down in alternating colors and descending order, building Foundations up, and strategically moving cards to reveal face-down ones, utilize the Waste Pile, and advance cards to Foundations.

Questions, generated from the current card layout, cover diverse Klondike decision-making and analysis scenarios. Types include identifying valid moves, determining the most effective move strategy (e.g., one that reveals a card or helps build Foundations), and analyzing for deadlocks. Players must apply logical reasoning based on on-screen card information and Klondike rules to select correct answers. Difficulty is dynamically set by the number of face-up cards.

![](images/8fd168f815ea56aac9663f1e600560136fd10318f2a5a57d425b922205a85461.jpg)  
Easy   
face-up cards $\leq 19$   
Medium   
face-up cards∈ [20, 23]   
Hard   
face-up cards $\geq 24$

![](images/5a65bd82d06249ed70630d254e234b5d456762f23a9b810336519a019a88853d.jpg)

![](images/55424980e3b8ff1263fe243b444a200c622769d05dc326d3541159edb69d48a1.jpg)

# J.3 Multi-step Reasoning

# J.3.1 Star Battle

This scene presents a $2\mathrm{D}$ $\mathfrak{n}\times \mathfrak{n}$ matrix which are divided into n regions. Each region has a specified color and is connective. The goal is to place stars in the matrix to make sure each row, col, region has only one star and the stars must not be adjacent to each other on rows, columns and diagonals. Complexity is controlled by adjusting the matrix size, testing the model's ability to reason according to the known rules.

![](images/fb05ae677717151b756f732007eb572255dc7c8c058558c9b068c18e5fcea31e.jpg)  
Images and Plot Level division

Question information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Identify the cell belonging to the given region</td></tr><tr><td>Q2</td><td>Target Perception</td><td>Easy</td><td>Identify the cell belonging to the given region and containing a star</td></tr><tr><td>Q3</td><td>State Prediction</td><td>Medium</td><td>Identify the cell where a star can be placed</td></tr><tr><td>Q4</td><td>State Prediction</td><td>Hard</td><td>Find the position of the final star needed to complete the puzzle</td></tr></table>

# Specific questions and analysis

# Introduction:

We have a $5^{*}5$ grid. The grid is divided into 5 regions. Cells with the same color belong to the same region.

Colors: Region0 (light pink), Region1 (powder blue), Region2 (light green), Region3 (peach), Region4 (red), Region5 (yellow), Region6 (cyan), Region7 (orange).

In the image, a star is represented by a black dot. If a cell has been placed a star, a black dot will be shown on this cell. We should place the star in this Star Battle Puzzle according to the following rules:

Each row must contain exactly 1 star(s). Each column must contain 1 star(s). Each region must contain exactly 1 star(s). Stars cannot be adjacent to each other, including diagonally.

The cells in the grid are labeled with row and column numbers starting from 0. The top-left corner of the grid is $(0,0)$ . $(x,y)$ means a cell at row $x$ and column $y$ . Now we have placed some stars in the grid.

Q1 (E2): The region with index 1 is represented by the color powder blue in the grid. Given the current state, which cell in the following options belong to region 1?

Options:   
1. (2, 3); 2. (0, 2); 3. (3, 4); 4. (3, 1);   
5. (2, 2); 6. (3, 2); 7. (2, 1); 8. (1, 0)

Analysis: The region with index 1 is represented by the color powder blue in the grid. In this puzzle, we need to identify which cell in the following options belongs to this region. The region 1 contains the following cells: $(0,0)$ , $(0,1)$ , $(0,2)$ , $(1,1)$ , $(1,2)$ . So $(0,2)$ belongs to region 1. The answer is Option 2.

Q2 (E2): In the current puzzle state, region 1 is associated with color powder blue. Please identify which of the following cells in this region that contains a star?

Options:

1. (1, 1); 2. (2, 0); 3. (0, 1); 4. (0, 0);   
5. (0, 2); 6. (4, 4); 7. (1, 2); 8. (1, 4)

Analysis: In this task, we need to find all the stars in the region with index 1. The region with index 1 corresponds to the color powder blue. This region contains the following cells: $(0,0)$ , $(0,1)$ , $(0,2)$ , $(1,1)$ , $(1,2)$ . Note that a star is represented by a black dot. Now scan the cells of the region 1 on the image. The cell with a black dot is: $(0,1)$ . So the answer is Option 3.

Q3 (M1): Now we have placed some stars in the grid. Based on the current puzzle state, which of the following cells can a star be placed in?

Options:

1. (4, 2); 2. (5, 3); 3. (1, 1); 4. (2, 2);   
5. (1, 0); 6. (3, 4); 7. (3, 3); 8. (4, 0)

Analysis: Cell (3, 3) cannot hold a star because: It is adjacent to a star, so it cannot hold a star. Cell (4, 2) cannot hold a star because: It is adjacent to a star, so it cannot hold a star. Besides, this cell is in region 2, which already contains one star, so it cannot hold a star. Cell (1, 0) cannot hold a star because: It is not adjacent to any star. However, this cell is in region 3, which already contains one star, so it cannot hold a star. Cell (2, 2) cannot hold a star because: It is adjacent to a star, so it cannot hold a star. Cell (5, 3) cannot hold a star because: It is not adjacent to any star. This cell is in region 1, which contains no stars. However, Column 3 has already been placed a star. Therefore, it cannot hold a star. Cell (1, 1) cannot hold a star because: It is not adjacent to any star. However, this cell is in region 3, which already contains one star, so it cannot hold a star. Cell (3, 4) cannot hold a star because: It is not adjacent to any star. This cell is in region 0, which contains no stars. However, Row 3 has already been placed a star. Therefore, it cannot hold a star. Cell (4, 0) can hold a star because: It is not adjacent to any star. This cell is in region 4, which contains no stars. Both row 4 and column 0 now have no stars. Thus, the correct answer is Option 8.

Q4 (H1): Now the puzzle has only one star left to be placed. The left star should be placed in which cell?

Analysis: **Step-by-step reasoning to solve the puzzle:**

1. **Preplaced stars and their positions:**

- The following stars are already placed: Row 1, Column 0, Row 2, Column 3, Row 3, Column 7, Row 4, Column 5, Row 5, Column 1, Row 6, Column 4, Row 7, Column 6.

- These positions fulfill the requirement of placing one star per row, column, and region.

2. **Identify rows and columns with and without stars:**

- **Rows with stars:** Row 1, Row 2, Row 3, Row 4, Row 5, Row 6, Row 7.

- **Rows without stars:**** Row 0.

- **Columns with stars:** Column 0, Column 1, Column 3, Column 4, Column 5, Column 6, Column 7.

- $^{**}$ Columns without stars:\*Column 2.

3. **Determine remaining valid cell:**

- The final star must be placed in a row and column that are both missing stars.

- Based on the information above, the row without a star is Row 0 and the column without a star is Column 2.

- The only available intersection is cell $(0,2)$ , which satisfies the row and column constraints.

4. **Region check:**

- The preplaced stars occupy the following regions: 0, 1, 2, 3, 4, 5, 6.

- The remaining region that requires a star is: Region 7.

5. **Final validation:**

- The cell $(0,2)$ belongs to the remaining region without a star. - Placing the star here satisfies all row, column, region, and adjacency constraints.

Thus, the final star must be placed at $^{**}$ Row 0, Column $2^{**}$ .

# J.3.2 Suku

Sudoku is a puzzle that requires filling a grid such that each row, column, and subgrid contains all digits from 1 to 9 without repetition. Our Sudoku-like puzzle game can be adapted to serve as a multi-modal dataset by replacing the numbers 1-9 with nine different colors. In this game, players are provided with a grid, where each row, column, and subgrid must contain all nine colors without repetition.

The types of questions in the game are as follows: 1. The color of a specific cell. 2. The number of cells of a certain color on the board. 3. The number of rows, columns, or blocks with more blank cells than a specified number. 4. The number of possible color options for a specific cell under the current board conditions. 5. The color for a third cell after two other cells are filled with specific colors. The difficulty level of the game is determined by the size of the grid and the number of filled cells.

Images and Plot Level division

![](images/a4503ba563122e769e679e4902f3ec0b9f987fd08179e6aa1156c15d8fbd0576.jpg)  
Easy   
$4\times 4$   
Medium   
$9 \times 9$ , most cells filled   
Hard   
$9 \times 9$ , fewer cells filled   
E1

![](images/8c7d5a82ecc85627a0afba7662b0c23e95288c785c81c1abdd0cb46c96593c83.jpg)  
M1

![](images/fe5c8556cad9c23d37c8674eb8a030f7d285ec5c95cf37f53913fe742b5d447c.jpg)  
H1

![](images/08ccfdcc2f255355ce8ee0b1a8a394b8d38dc5f26718b502dc59b1c0485b861a.jpg)  
E2

![](images/9a1444c3412ae35dd521fe3cfa95476649352914605b1b616b2eff16197f6617.jpg)  
M2

![](images/17a302aea6c2ff55608956f7a9e66406ea5450e0c9175a397719b2377b172fd5.jpg)  
H2

Question information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Position color identification</td></tr><tr><td>Q2</td><td>Target Perception</td><td>Easy</td><td>Color occurrence count</td></tr><tr><td>Q3</td><td>Target Perception</td><td>Medium</td><td>Sparse unit count</td></tr><tr><td>Q4</td><td>State Prediction</td><td>Medium</td><td>Valid color candidated inference</td></tr><tr><td>Q5</td><td>State Prediction</td><td>Hard</td><td>Guided-position color deduction</td></tr></table>

# Specific questions and analysis

Q1 (E1): What color is at position $(2,1)$ (note that on the board the position $(2,1)$ has already been filled with a certain color)? Choose from the following options: A.red, B.green, C.blue, D.magenta

Analysis: From the image, we can see the color at Position $(2,1)$ is red. So the answer is A.

Q2 (M1): How many times does aqua appear on the board?

Analysis: Color aqua appears at: (1,1), (2,9), (3,6), (4,7), (5,3), (6,4), (7,8), (8,2), (9,5), total 9 times. So the answer is 9.

Q3 (E2): How many columns have more than 1 empty cell?

Analysis: Col analysis:

col 1 has 3 empty cells in positions 1, 2, 3;

col 2 has 2 empty cells in positions 1, 4;

col 3 has 1 empty cells in positions 2;

col 4 has 2 empty cells in positions 1, 4.

In total, 3 col(s) have more than 1 empty cell.

So the answer is 3.

Q4 (H1): How many colors can be filled in position (7,1)? Infer based on the current situation focusing only on the colour of the position.

Analysis: Constraint analysis for position (7,1):

Existing colors in row: purple, aqua, forest green, gray, yellow, red, green

Existing colors in column: blue, aqua, purple, red, green, yellow, forest green, purple

Existing colors in box: purple, aqua, green, yellow, gray, blue

Therefore, possible colors are: magenta. So the answer is 1.

Q5 (H2): After determining colors at positions (2,1), (2,5), what color should be at position (2,4)? Choose from following options: A.red, B.green, C.blue, D.magenta, E.yellow, F.aqua, G(gray, H.purple, I_forest green

Analysis: Deductive reasoning process:

Step 1: Position (2,1): Existing colors in the row: green, aqua, gray, purple, blue, red. Existing colors in the column: purple, blue, aqua, gray, yellow, magenta. Existing colors in the 3x3 box: purple, gray, magenta, green, aqua, yellow, blue

Therefore, the only possible color for this position is forest green.

Step 2: ... Therefore, the only possible color for this position is magenta.

Final analysis for position (2,4): ... After previous deductions, possible color reduced to: yellow So the answer is E.

# J.3.3 Langton's Ant

This game simulates the behavior of Langton's Ant in a cellular automaton. The ant is represented by a red arrow indicating its initial position and direction. It moves on a randomly generated grid composed of black and white squares, following a fixed set of rules: If the ant is on a black square, it turns 90 degrees to the right, flips the square to white, and moves forward one step; if the ant is on a white square, it turns 90 degrees to the left, flips the square to black, and moves forward one step.

There are three types of questions in the game: 1. Identify the ant's initial position and direction. 2. Predict the ant's position and direction after a given number of steps. 3. Given a specific square, infer how many times its color has changed after the ant has moved a certain number of steps.

The difficulty of the game is determined by the question type and the size of the grid: the three question types increase in complexity respectively, and the grid size defines the level of difficulty— $n = 5$ indicates an easy level, while $n = 13$ indicates a hard level.

Images and Plot Level division

![](images/38e158d57f41f318fcda7bbbc885297f24230536bd8d04a6c7d9430bc89d1bbf.jpg)  
Easy   
$5 \times 5$   
Medium   
$9 \times 9$   
Hard   
$13\times 13$   
E

![](images/773355f3353ce2d2bfdad06e0de39daa5b7c0d2725d4900ec50dc54bb52ae35b.jpg)  
M

![](images/2eaba7888829ccc0208c751ccc74528c27fd9419956627bd3f16faf3367ab9fc.jpg)  
H

Question information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Identify the current position and direction of the ant.</td></tr><tr><td>Q2</td><td>State Prediction</td><td>Medium</td><td>Predict the ant&#x27;s position and direction after several steps.</td></tr><tr><td>Q3</td><td>State Prediction</td><td>Hard</td><td>Count how many times a specific cell changes its color.</td></tr></table>

# Specific questions and analysis

# Introduction:

In Langton's Ant, we have a grid where each cell is either white or black. A red arrow represents an ant, showing its current position and direction. The ant follows these simple rules:

1. If the ant is on a white cell, it turns right 90 degrees, changes the cell to black, and moves forward one step   
2. If the ant is on a black cell, it turns left 90 degrees, changes the cell to white, and moves forward one step   
3. If the ant would move off the grid, it wraps around to the opposite side (using modulo with grid size)

Q1 (E): What is the current position and direction of the ant in the image?

Answer using one of the following options with its corresponding letter:

A: Position (1, 3), facing up; B: Position (0, 4), facing left   
C: Position (2, 3), facing down; D: Position (4, 3), facing up   
E: Position (4, 3), facing down; F: Position (0, 0), facing up   
G: Position $(0, 0)$ , facing right; H: Position $(4, 3)$ , facing right

Analysis: Step-by-step analysis:

1. Look at the red arrow in the image which represents the ant. 2. The arrow's position indicates the ant is at coordinates $(0,0)$ . 3. The arrow's direction shows the ant is facing right.

Therefore, the ant's current position is $(0, 0)$ and it's facing right. The answer is G.

Q2 (E): After 6 steps, what will be the ant's position and direction?

Answer using one of the following options with its corresponding letter:

A: Position (2, 0), facing down; B: Position (2, 3), facing right   
C: Position $(0,0)$ , facing right; D: Position $(4,1)$ , facing right   
E: Position (4, 4), facing up; F: Position (0, 3), facing down   
G: Position (2, 1), facing left; H: Position (4, 4), facing left

Analysis: Initial state: The ant is at $(0,0)$ facing right.

Let's follow the ant's movement step by step:

- Step 1: Ant is on a white cell at $(0,0)$ , facing right. It turns right, changes the cell to black, moves forward to $(1,0)$ , now facing down.   
- Step 2: Ant is on a white cell at $(1,0)$ , facing down. It turns right, changes the cell to black, moves forward to $(1,4)$ , now facing left.   
- Step 3: Ant is on a white cell at (1, 4), facing left. It turns right, changes the cell to black, moves forward to (0, 4), now facing up.   
- Step 4: Ant is on a white cell at $(0, 4)$ , facing up. It turns right, changes the cell to black, moves forward to $(0, 0)$ , now facing right.   
- Step 5: Ant is on a black cell at $(0,0)$ , facing right. It turns left, changes the cell to white, moves forward to $(4,0)$ , now facing up.   
- Step 6: Ant is on a white cell at $(4,0)$ , facing up. It turns right, changes the cell to black, moves forward to $(4,1)$ , now facing right.

Final state: The ant is at (4, 1) facing right. The answer is D.

Q3 (E): After 12 steps, how many times did the cell at position $(0,0)$ change its color? Answer with a number.

Analysis: Initial state: The ant is at $(0,0)$ facing right. Target cell $(0,0)$ starts as white.

Let's follow the ant's movement step by step:

- Step 1: Ant is on a white cell at $(0,0)$ , facing right. It turns right, changes the cell to black, moves forward to $(1,0)$ , now facing down. Target cell $(0,0)$ is visited and changes from white to black (change #1).   
- Step 2: Ant is on a white cell at $(1,0)$ , facing down. It turns right, changes the cell to black, moves forward to $(1,4)$ , now facing left. Target cell $(0,0)$ remains black.   
- Step 3: Ant is on a white cell at (1, 4), facing left. It turns right, changes the cell to black, moves forward to (0, 4), now facing up. ...

Step 4: Ant is on a white cell at $(0, 4)$ , facing up. It turns right, changes the cell to black, moves forward to $(0, 0)$ , now facing right. ...

Step 5: Ant is on a black cell at $(0,0)$ , facing right. It turns left, changes the cell to white, moves forward to $(4,0)$ , now facing up. Target cell $(0,0)$ changes from black to white (change $\#2$ ).   
- Step 6: Ant is on a white cell at $(4,0)$ , facing up. It turns right, changes the cell to black, moves forward to $(4,1)$ , now facing right. ...   
- Step 7: Ant is on a white cell at (4, 1), facing right. It turns right, changes the cell to black, moves forward to (0, 1), now facing down. ...

... (Omitted: Step 8-11. Ant continues moving, flipping cells, but $(0,0)$ remains white.)

- Step 12: Ant is on a white cell at $(0,1)$ , facing down. It turns right, changes the cell to black, moves forward to $(0,0)$ , now facing left. Target cell $(0,0)$ remains white.

Final state: The ant is at $(0, 0)$ facing left. Target cell $(0, 0)$ changed color 2 times. The answer is 2.

# J.3.4 Word Search

This game is a visual search task based on the classic Word Search puzzle paradigm. It features a grid where each cell contains a single letter. Target words are embedded within this grid, oriented horizontally, vertically, or diagonally (spanning eight possible directions).

Question types assess visual parsing and pattern recognition within the grid, including: (1) identifying the

letter located at a specific row and column index; (2) counting the total occurrences of a given letter across the entire grid; (3) determining the direction (out of eight possibilities) in which a specified word extends, given its starting cell coordinates; and (4) locating both the starting cell coordinates and the correct direction for a given target word within the grid. The complexity ('plot level') is influenced by the grid size.

<table><tr><td colspan="6">Easy
5×5</td><td colspan="8">Medium
7×7</td><td colspan="7">Hard
8×8</td><td></td><td></td><td></td></tr><tr><td></td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td></td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td></td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td></td></tr><tr><td>1</td><td>B</td><td>M</td><td>H</td><td>J</td><td>X</td><td></td><td>T</td><td>O</td><td>Z</td><td>X</td><td>D</td><td>T</td><td>N</td><td></td><td>D</td><td>T</td><td>O</td><td>D</td><td>C</td><td>J</td><td>C</td><td>C</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2</td><td>B</td><td>R</td><td>U</td><td>I</td><td>J</td><td>P</td><td>Z</td><td></td><td>U</td><td>L</td><td>V</td><td>G</td><td>H</td><td>G</td><td>A</td><td>G</td></tr><tr><td>2</td><td>Z</td><td>C</td><td>C</td><td>P</td><td>H</td><td></td><td>3</td><td>Y</td><td>K</td><td>S</td><td>G</td><td>Z</td><td>Q</td><td>O</td><td></td><td>Z</td><td>A</td><td>F</td><td>U</td><td>A</td><td>J</td><td>M</td><td>I</td></tr><tr><td>3</td><td>A</td><td>I</td><td>X</td><td>T</td><td>N</td><td></td><td>4</td><td>O</td><td>Z</td><td>B</td><td>X</td><td>B</td><td>A</td><td>S</td><td></td><td>P</td><td>K</td><td>X</td><td>L</td><td>R</td><td>W</td><td>T</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5</td><td>X</td><td>X</td><td>M</td><td>V</td><td>T</td><td>R</td><td>S</td><td></td><td>O</td><td>E</td><td>P</td><td>V</td><td>V</td><td>C</td><td>D</td><td>K</td></tr><tr><td>4</td><td>F</td><td>J</td><td>G</td><td>O</td><td>O</td><td></td><td>6</td><td>R</td><td>L</td><td>W</td><td>P</td><td>Q</td><td>R</td><td>P</td><td></td><td>G</td><td>P</td><td>G</td><td>Y</td><td>A</td><td>M</td><td>C</td><td>J</td></tr><tr><td>5</td><td>W</td><td>O</td><td>Q</td><td>M</td><td>P</td><td></td><td>7</td><td>S</td><td>C</td><td>D</td><td>W</td><td>C</td><td>X</td><td>T</td><td></td><td>F</td><td>N</td><td>S</td><td>E</td><td>H</td><td>J</td><td>J</td><td>Z</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8</td><td>T</td><td>R</td><td>Y</td><td>O</td><td>S</td><td>K</td><td>C</td><td>X</td></tr></table>

# J.3.5 2D Turing Machine

This game presents a simulation of a two-dimensional Turing machine. The state of the machine's tape, represented as a grid, is visualized using distinct colors for different symbols within each cell. The initial position of the read/write head is indicated visually, typically by a black dot, and is also specified in the accompanying text description. The core task requires simulating the step-by-step operation of the defined Turing machine. Question types focus on tracking the machine's execution, including: (1) determining the head's coordinates after a specified number of steps; (2) identifying the symbol (color) under the head after a specified number of steps; (3) describing the sequence of symbol changes within a particular cell over a given number of steps; and (4) identifying the step number at which the machine first enters a specific state. The complexity ("Plot Level") of the task is primarily determined by the dimensions of the grid.

![](images/92e9dec3107681cfbd5ee8efc3dd23366d943d051f4b9b453eefa6b957c3c65f.jpg)

# J.3.6 Tents

Tents is a logic puzzle played on a grid with predefined tree positions and row/column tent counts. The objective is to place tents adjacent to trees while following these rules: each cell holds either a tree, a tent, or remains empty; the number of tents matches the number of trees; every tent must be horizontally or vertically adjacent to at least one tree; no two tents can be adjacent in any direction (including diagonally); and row/column tent totals must match the given numbers.

Questions involve analyzing partially filled grids, such as determining the current number of tents in a row, remaining tents to place, identifying tree locations among given positions, available spots for new tents without immediate rule violations, and selecting rule-compliant tent placements. Puzzle difficulty scales with grid size.

<table><tr><td colspan="9">Easy
7×7</td><td colspan="9">Medium
10×10</td><td colspan="9">Hard
13×13</td><td></td></tr><tr><td>2
0</td><td>0
1</td><td>0
2</td><td>2
3</td><td>0
4</td><td>0
5</td><td>1
6</td><td></td><td></td><td>1
0</td><td>2
1</td><td>1
2</td><td>1
3</td><td>2
4</td><td>1
5</td><td>2
6</td><td>0
7</td><td>0
8</td><td>0
9</td><td>0
1
0</td><td>2
1
2</td><td>0
3
4</td><td>2
5
6</td><td>1
7
7</td><td>0
8
7</td><td>1
9
8</td><td>0
10
9</td><td>0
11
12</td></tr><tr><td rowspan="6">0
3
2
0
3
0
4
2
5
0
6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0
1
2
3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0
2
3
4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0
4
5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0
6
7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0
8
9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

# J.3.7 Rhythm Game

This is a rhythm game featuring dynamic falling blocks. Players are tasked with selecting a column to place their finger and clicking on the operation blocks that fall to the first row of the selected column to score points. Alternatively, players may choose not to click any column, which will not affect the falling of the blocks. The blocks in the game are divided into three types: Click blocks, Reverse blocks, and Snake blocks, each with different scores and click effects, prompting players to make choices while playing to get the highest score.

Questions will be asked based on the current game situation, involving issues such as block type identification, grid ratio calculation, and score calculation. Players need to reason and answer based on the information on the screen in the game. In addition, the game difficulty is divided into three levels according to the complexity of the scene, including Easy $(15 \times 4)$ , Medium $(15 \times 6)$ , and Hard $(20 \times 6)$ .

<table><tr><td colspan="4">Easy
15 × 4</td><td colspan="4">Medium
15 × 6</td><td colspan="4">Hard
20 × 6</td></tr><tr><td>15</td><td></td><td></td><td></td><td>15</td><td></td><td></td><td></td><td></td><td>20</td><td></td><td></td></tr><tr><td>14</td><td></td><td></td><td></td><td>14</td><td></td><td></td><td></td><td></td><td>19</td><td></td><td></td></tr><tr><td>13</td><td></td><td></td><td></td><td>13</td><td></td><td></td><td></td><td></td><td>18</td><td></td><td></td></tr><tr><td>12</td><td></td><td></td><td></td><td>12</td><td></td><td></td><td></td><td></td><td>17</td><td></td><td></td></tr><tr><td>11</td><td></td><td></td><td></td><td>11</td><td></td><td></td><td></td><td></td><td>16</td><td></td><td></td></tr><tr><td>10</td><td></td><td></td><td></td><td>10</td><td></td><td></td><td></td><td></td><td>15</td><td></td><td></td></tr><tr><td>9</td><td></td><td></td><td></td><td>9</td><td></td><td></td><td></td><td></td><td>14</td><td></td><td></td></tr><tr><td>8</td><td></td><td></td><td></td><td>8</td><td></td><td></td><td></td><td></td><td>13</td><td></td><td></td></tr><tr><td>7</td><td></td><td></td><td></td><td>7</td><td></td><td></td><td></td><td></td><td>12</td><td></td><td></td></tr><tr><td>6</td><td></td><td></td><td></td><td>6</td><td></td><td></td><td></td><td></td><td>11</td><td></td><td></td></tr><tr><td>5</td><td></td><td></td><td></td><td>5</td><td></td><td></td><td></td><td></td><td>10</td><td></td><td></td></tr><tr><td>4</td><td></td><td></td><td></td><td>4</td><td></td><td></td><td></td><td></td><td>9</td><td></td><td></td></tr><tr><td>3</td><td></td><td></td><td></td><td>3</td><td></td><td></td><td></td><td></td><td>8</td><td></td><td></td></tr><tr><td>2</td><td></td><td></td><td></td><td>2</td><td></td><td></td><td></td><td></td><td>7</td><td></td><td></td></tr><tr><td>1</td><td></td><td></td><td></td><td>1</td><td></td><td></td><td></td><td></td><td>6</td><td></td><td></td></tr><tr><td></td><td>1</td><td>2</td><td>3</td><td>4</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td></td></tr></table>

# J.3.8 Lifegame

This is a cellular automaton simulation on an $n \times n$ 2D grid, where cells are either "alive" (black squares) or "dead" (white/empty squares) and evolve over generations. A cell's next state is determined by its current state and its eight neighbors: a dead cell with exactly three live neighbors becomes alive (simulating reproduction); an alive cell dies with fewer than two (simulating underpopulation) or more than three live neighbors (simulating overpopulation), but survives with two or three.

Game tasks involve counting current alive cells, predicting alive cells after one iteration, determining a specific cell's state change over N iterations, and calculating iterations for a given region to reach a stable state (static or oscillating). The "Plot Level" (difficulty) is determined by the grid size, with larger grids indicating higher difficulty.

![](images/d5354e579eeccb6c4d1d38f415da35662b335581fab22f585f0777509eabd3c7.jpg)  
Easy   
$3\times 3$   
Medium   
$4\times 4$   
Hard   
$5 \times 5$

![](images/fabc94ed2512cf397df41430f14b3dcefcf242656bcb41880d06b2f06eaf2238.jpg)

![](images/3898bdae33720f905bed6330898effb3f83d71b5a02d74cfe8d8a593b605a467.jpg)

# J.3.9 Minesweeper

The game is inspired by Microsoft's classic game Minesweeper. The objective is to reveal all cells that do not contain mines while correctly flagging the mines. If a player accidentally reveals a cell containing a mine, the game ends immediately. The Minesweeper game board consists of cells marked with numbers (indicating the number of mines in the surrounding 3x3 grid), white revealed cells, gray hidden cells, flagged cells (marked with the letter "F"), and cells containing mines, which are unknown to the player. The game tasks include determining the status of cells, inferring the locations of mines, predicting the outcome of actions, and deciding on optimal reveal strategies. The difficulty levels are determined by the board size, with 4x4 being easy, 5x5 being medium, and 6x6 being hard. The board size and the number of mines change based on the difficulty level.

![](images/0648e973f4c21f195a6604a25239b5cab84bdc7f384c4df3bead6393a61b4d6c.jpg)  
Easy   
$4\times 4$   
Medium   
$5\times 5$   
Hard   
$6\times 6$   
Minesweeper Board

![](images/bd739288bda70899a44139a47b2965b8cd7cd5d5586d1d9f103780619d499c58.jpg)  
Minesweeper Board

![](images/4834b217da36f2797c1932f5c19076d390e01aaf4f7d99c0af16e17dd759fa56.jpg)  
Minesweeper Board

# J.4 Strategic Planning

# J.4.1 Sokoban

This game is based on the classic Sokoban puzzle game. The game scene consists of a grid-based area featuring a player (represented by a black humanoid figure), boxes (brown squares with X texture), target points (green X marks), walls (brick-textured barriers), and movable areas (light brown floor). Players can move in four directions (up, down, left, right), push boxes forward, but cannot pull boxes or move through walls. The objective is to push all boxes onto target points. Question types evaluate spatial planning and logical reasoning: (1) predicting the player's final position after a sequence of moves; (2) predicting a box's final position after a sequence of movements; (3) determining the minimum number of moves required to solve the puzzle; (4) identifying the current position of the player; (5) calculating the Manhattan distance between a box and its target point; and (6) finding the optimal sequence of moves to reach a specific position. The game difficulty is determined by the board size.

Problem information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Identify the current position of the player on the board</td></tr><tr><td>Q2</td><td>Target Perception</td><td>Easy</td><td>Calculate the Manhattan distance between a box and its target</td></tr><tr><td>Q3</td><td>State Prediction</td><td>Medium</td><td>Given a sequence of player moves, predict the final position of the player</td></tr><tr><td>Q4</td><td>State Prediction</td><td>Medium</td><td>Given a sequence of moves, predict the final position of the box</td></tr><tr><td>Q5</td><td>Strategy Optimization</td><td>Hard</td><td>Find the optimal sequence of moves to reach a specific position</td></tr><tr><td>Q6</td><td>Strategy Optimization</td><td>Hard</td><td>Determine the minimum number of moves needed to solve the puzzle</td></tr></table>

![](images/af2fdc3ca397fb9175e545cea78bc9a7e929ee56470472fc5dd4af0abba18e8b.jpg)  
Images and Plot Level division

# Specific questions and analysis

Introduction: This is a Sokoban puzzle where cartoon person is player, green X is target, brown box with X is box to push, brown tiles are walls, and light brown areas are movable spaces. The coordinates $(\mathbf{x},\mathbf{y})$ in this puzzle represent the matrix format.

Q1 (M): What is the current position of the player (row, column)?

Options:   
[1] (6, 4) [2] (1, 4) [3] (3, 4) [4] (1, 5)   
[5] (4, 3) [6] (4, 1) [7] (5, 1) [8] (6, 3)

Analysis: - Player position: (1, 5) - Boxes positions: (4, 3) - Target positions: (5, 2) The player is currently at position (1, 5). So the answer is (1, 5). The option number is 4.

Q2 (M): What is the Manhattan distance between the box and the target?

Options:

[1] 15 [2] 16 [3] 6 [4] 2   
[5] 12 [6] 1 [7] 14 [8] 5

Analysis: - Player position: (1, 5) - Boxes positions: (4, 3) - Target positions: (5, 2) Box position: (4, 3) Target position: (5, 2) Manhattan distance $= |4 - 5| + |3 - 2| = 2$ . So the answer is 2. The option number is 4.

Q3 (E): If the player makes these moves: $\mathrm{Up} \to \mathrm{Down} \to \mathrm{Left} \to \mathrm{Up} \to \mathrm{Down} \to \mathrm{Left} \to \mathrm{Left} \to \mathrm{Up}$ , where will player end up?

Options:

[1] (6, 5) [2] (1, 6) [3] (6, 6) [4] (2, 3)   
[5] (2, 6) [6] (1, 2) [7] (5, 2) [8] (2, 5)

Analysis: - Player position: (1, 5) - Boxes positions: (4, 3) - Target positions: (5, 2) Move sequence analysis: Initial position: (1, 5) Move 1 - Up: Failed - Wall in the way (Player stays at (1, 5)) Move 2 - Down: Player moves from (1, 5) to (2, 5) Move 3 - Left: Player moves from (2, 5) to (2, 4) Move 4 - Up: Player moves from (2, 4) to (1, 4) Move 5 - Down: Player moves from (1, 4) to (2, 4) Move 6 - Left: Player moves from (2, 4) to (2, 3) Move 7 - Left: Player moves from (2, 3) to (2, 2) Move 8 - Up: Player moves from (2, 2) to (1, 2) Final position: (1, 2). So the answer is (1, 2). The option number is 6.

Q4 (M): Treat boxes as objects that can move by themselves, and treat people as floor (movable areas). After the moves up, right, down, up, left, right, up, left, where will the box that started at position (4, 3) end up?

Options:

[1] (2, 3) [2] (3, 6) [3] (3, 1) [4] (1, 5)   
[5] (6, 2) [6] (4, 6) [7] (3, 5) [8] (6, 4)

Analysis: - Player position: (1, 5) - Boxes positions: (4, 3) - Target positions: (5, 2) Move sequence: Move up: Box moved from (4, 3) to (3, 3) Move right: Box moved from (3, 3) to (3, 4) Move down: Box moved from (3, 4) to (4, 4) Move up: Box moved from (4, 4) to (3, 4) Move left: Box moved from (3, 4) to (3, 3) Move right: Box moved from (3, 3) to (3, 4) Move up: Box moved from (3, 4) to (2, 4) Move left: Box moved from (2, 4) to (2, 3) Box moves from (4, 3) to (2, 3). So the answer is (2, 3). The option number is 1.

Q5 (M): Treat the boxes as walls. What is the shortest sequence of moves for human to move himself from position (1, 5) to position (1, 6)?

Options:

[1] Down [2] Left [3] Down $\rightarrow$ Right $\rightarrow$ Down [4] Right   
[5] Right $\rightarrow$ Down $\rightarrow$ Left [6] Down $\rightarrow$ Right $\rightarrow$ Left   
[7] Down $\rightarrow$ Left $\rightarrow$ Up [8] Left $\rightarrow$ Down

Analysis: - Player position: (1, 5) - Boxes positions: (4, 3) - Target positions: (5, 2) Start position: (1, 5) End position: (1, 6) Optimal move sequence: Right. So the answer is Right. The option number is 4.

Q6 (M): What is the minimum number of moves needed to solve this puzzle?

Options:   
[1] 5 [2] 15 [3] 10 [4] 7   
[5] 11 [6] 9 [7] 6 [8] 8

Analysis: - Player position: (1, 5) - Boxes positions: (4, 3) - Target positions: (5, 2) Solution analysis: Step-by-step solution: Player moves from (1, 5) to (2, 5) Player moves from (2, 5) to (3, 5) Player moves from (3, 5) to (3, 4) Player moves from (3, 4) to (3, 3) Player moves from (3, 3) to (4, 3) (box moves from (4, 3) to (5, 3)) Player moves from (4, 3) to (4, 4) Player moves from (4, 4) to (5, 4) Player moves from (5, 4) to (5, 3) (box moves from (5, 3) to (5, 2) Total player moves: 8. So the answer is 8. The option number is 8.

# J.4.2 Maze

This project focuses on generating question-and-answer datasets for a grid-based maze game. In this game, a player, represented by a red circle, must navigate a path of white blocks to reach a green goal block, while avoiding blue obstacle blocks. Movement is restricted to the four cardinal directions. The generated questions are designed to evaluate a range of cognitive abilities, primarily centered on spatial reasoning and pathfinding. These include tasks such as identifying the current locations of game elements, determining permissible moves, predicting the outcomes of specific actions, and deducing optimal routes to the goal. The complexity of the mazes and the associated questions scales, with mazes offered in Small, Medium, and Large sizes, and individual questions categorized by difficulty.

# Images and Plot Level division

![](images/b6b62dd8a94647b4b5cf8999f44ea8c7a88c911fff7b3bd642694e3299a2a25c.jpg)

# Problem information

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Ask the position of player</td></tr><tr><td>Q2</td><td>Target Perception</td><td>Easy</td><td>Ask the position of goal within the maze</td></tr><tr><td>Q3</td><td>Target Perception</td><td>Easy</td><td>Ask the available directions to move are currently</td></tr><tr><td>Q4</td><td>State Prediction</td><td>Medium</td><td>The position after moving</td></tr><tr><td>Q5</td><td>Strategy Optimization</td><td>Hard</td><td>Find the path to the goal</td></tr><tr><td>Q6</td><td>Strategy Optimization</td><td>Hard</td><td>Count how many turns it takes to reach the finish</td></tr></table>

# Specific questions and analysis

Introduction:

1. This is a maze mini-game. The player needs to navigate around obstacles to reach the destination and achieve victory.   
2. The red circle represents the player, the green block is the goal and the blue blocks are obstacles.   
3. The player can only move within the white blocks.   
4. The coordinates are given in the format (row, col), where row represents the vertical position and col represents the horizontal position.

Q1 (E): Which of the following are the coordinates of the player?

Options:

A. (4, 6); B. (5, 5); C. (3, 5); D. (4, 4); E. (4, 5)

Analysis: Take a look at the game screen, the red circle represents the player. The coordinates of player are (4, 5), so the right option is E.

Q2 (E): Which of the following are the coordinates of the goal?

Optoins:

A. (7, 7); B. (7, 8); C. (6, 7); D. (7, 6); E. (8, 7)

Analysis: Take a look at the game screen, the green block represents the goal. The coordinates of goal are (7, 7), so the right option is A.

Q3 (E): Which directions are available to move now?

Options:

A.up;B.down;C.up,down;D.up,right;   
E. left, right; F. up, down, right;   
G. down, left, right; H. up, down, left, right

Analysis: The player is on (4, 5), and (3, 5) (5, 5) is empty. The player can move up, down. Therefore, the option is C.

Q4 (E): What are the coordinates of player after moving down?

Options:

A. (4, 6)   
B. (5, 5)   
C. (3, 5)   
D. (4, 4)   
E. (4, 5)

Analysis: Observe the screen, the position of player is (4,5). After moving down, the player is in (5, 5).

Therefore, the right option is B.

Q5 (E): Which sequence of movements will allow the player to reach the destination?

Options:   
A. left, left, left, right, right, left   
B. down, right, right, down, down   
C. down, down, left, up, right, down   
D.up,up,up,right,up,down   
E. left, down, left, right, down, left

Analysis: Let's figure out the path to the goal step by step: Step 1. Go down, from (4, 5) to (5, 5). Step 2.

Go right, from (5, 5) to (5, 6). Step 3. Go right, from (5, 6) to (5, 7). Step 4. Go down, from (5,

7) to (6, 7). Step 5. Go down, from (6, 7) to (7, 7). Achieved the goal! Therefore, the right

sequence of movements are: down, right, right, down, down. The right option is B.

Q6 (E): Find the path to the finish and count the number of turns it takes to get there. Provide one number.

Analysis: First, let's figure out the path to the goal step by step: Step 1. Go down, from (4, 5) to (5, 5). Step 2. Go right, from (5, 5) to (5, 6). Step 3. Go right, from (5, 6) to (5, 7). Step 4. Go down, from (5, 7) to (6, 7). Step 5. Go down, from (6, 7) to (7, 7). Achieved the goal! Therefore, the path is: (4, 5), (5, 5), (5, 6), (5, 7), (6, 7), (7, 7). Then, let's count the number of turns step by step: Step 2. Turn detected: from down to right. Step 3. No turn detected. Step 4. Turn detected: from right to down. Step 5. No turn detected. In summary, the total number of turns is 2.

# J.4.3 TicTacToe

This game is derived from the classic Tic-Tac-Toe game, featuring a $3 \times 3$ grid area with two players represented by red and blue grid markers respectively. The objective is to create a straight line of three same-colored markers either horizontally, vertically, or diagonally to win. Question types include: (1) determining the color of specific grid cells, (2) identifying the optimal move for the current player, and (3) predicting the opponent's best response after a given move. The difficulty scales across three levels based on scenario complexity, where higher difficulty requires evaluating progressively more decision-making conditions to answer the same question types, systematically testing the model's strategic reasoning and conditional judgment capabilities.

![](images/fbf3188ae5c9efdf93caf907137c372d264ca320927da86133bacb6c5a3dc66d.jpg)  
Images and Plot Level division   
Easy   
E

![](images/1446ab1878542e2a00144f62b2990b020e5fbb34a23de0cad6f6304c19ab8f95.jpg)  
Medium   
M

![](images/3e2ebe3163ebbadfd69e0f536e46ade753ee5817e03d61e250b0885bf5076035.jpg)  
Hard   
H

Question information   

<table><tr><td></td><td>QA type</td><td>QA Level</td><td>Description</td></tr><tr><td>Q1</td><td>Target Perception</td><td>Easy</td><td>Questions about the current state of a specific block of the board.</td></tr><tr><td>Q2</td><td>Strategy Optimization</td><td>Medium</td><td>Questions about the optimal strategy to take a move of the current player of the board.</td></tr><tr><td>Q3</td><td>Strategy Optimization</td><td>Hard</td><td>Questions about the outcome to take a specific move of the current player of the board, and the optimal strategy to take a move of the opponent player after the specific move.</td></tr></table>

# Specific questions and analysis

# Introduction:

Tic-Tac-Toe is a classic two-player game played on a $3 \times 3$ grid, (row, col) from $(0,0)$ to $(2,2)$ . Players take

turns marking a space in the grid, one using $^{**}\mathrm{O}^{**}$ (the red block) and the other using $^{**}\mathrm{X}^{**}$ (the blue block). In each game, player $^{**}\mathrm{O}^{**}$ starts first. The objective is to be the first to get three of your marks in a row (horizontally, vertically, or diagonally). If all nine squares are filled without either player achieving this, the game ends in a draw. Notice: the current player to make a move should be inferred from the number of pieces for each players on the board. When inferring the optimal move, if optimal move can be inferred by some rules, choose the optimal move. Otherwise, choose the first move. (The order of choices is $(0, 0)$ , $(0, 1)$ , $(0, 2)$ , $(1, 0)$ , ..., $(2, 2)$ , choose the first move that is not occupied)

Q1 (E): Question: What is the color of the block at $(0, 0)$ ?

Options: A. red; B. blue; C. white

Analysis: The current board is $\left[\left[\mathrm{O}^{\prime},\mathrm{O}^{\prime},\mathrm{X}^{\prime}\right],\left[\mathrm{X}^{\prime},\mathrm{X}^{\prime},\mathrm{O}^{\prime}\right],\left[\mathrm{,}^{\prime},\mathrm{O}^{\prime},\mathrm{X}^{\prime}\right]\right]$ . The block at $(0,0)$ is "O", and the color matching "O" is red, so the block at $(0,0)$ is red. The answer is A.

Q2 (M): What is the optimal move for the current player? If no move exists, choose the answer "None". Options: A. None; B. $(0, 0)$ ; C. $(0, 1)$ ; D. $(0, 2)$ ; E. $(1, 0)$ ; F. $(1, 1)$ ; G. $(1, 2)$ ; H. $(2, 0)$ or $(2, 1)$ or $(2, 2)$

Analysis: The current board is $\left[\left[{ }^{\prime }\right],\left.{ }^{\prime }\right.\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $\left.{ }^{\prime }\right.$ , $X^{\prime}$ , $\left.{ }^{\prime }\right.$ ]. Since the player "O" plays first in each game, if the count of "O" is the same as "X", the current player is "O". Otherwise, the current player is "X". The count of "O" is 2 and the count of "X" is 1, so the player now is X. Current player is X, opponent is O. Must block opponent O's potential double threat on Row 0 and Top-left to bottom-right diagonal, so player X should choose position (0, 0). The answer is B.

Q3 (H): If the current player moves to $(0,2)$ , will this move be successful? If not, choose the answer "None". If successful, will the current player win immediately? If yes, choose the answer "None". Otherwise, what is the opponent's optimal move following this step?

Options: A. None; B. (0, 0); C. (0, 1); D. (0, 2); E. (1, 0); F. (1, 1); G. (1, 2); H. (2, 0) or (2, 1) or (2, 2)

Analysis: Yes, this move will be successful. The current board is $\left[\left[{ }^{\prime }\right],\left.{ }^{\prime }\right],\left.{ }^{\prime }\right]\right],\left.\left[{ }^{\prime },{}^{\prime }\mathrm{X}^{\prime },{}^{\prime }\mathrm{O}^{\prime }\right],\left[{ }^{\prime },{}^{\prime },{}^{\prime },{}^{\prime }\right]\right]$ . Since the player "O" plays first in each game, if the count of "O" is the same as "X", the current player is "O". Otherwise, the current player is "X". The count of "O" is 1 and the count of "X" is 1, so the player now is O. Since the current player O moves to (0, 2), the current player won't win immediately. After that, current player is X, opponent is O. Must block opponent O's winning threat on Column 2, so player X should choose position (2, 2). The answer is H.

# J.4.4 Ultra TicTacToe

Ultra TicTacToe is an advanced variant of TicTacToe played on a 3x3 grid of 3x3 subgrids (Nine-grids). Players alternate placing "X" (first player) and "O" (second player) markers using a four-coordinate system (i,j,row,col), where (i,j) denotes the subgrid position and (row,col) specifies the cell within that subgrid. The initial move must be made in the central Nine-grid (2,2), with subsequent moves constrained to the subgrid determined by the opponent's previous move position. Scoring occurs when three identical markers form a line within any subgrid (each such line counts as 1 point). The game concludes when all nine central cells of the subgrids are occupied. Question types involve analyzing board states (identifying marker ownership at coordinates), calculating available move options, quantifying marked cells, evaluating scoring patterns within subgrids, and determining optimal strategic placements. Game complexity tiers are defined by move count ranges: Easy (10-34 steps), Medium (35-59 steps), and Hard (60-81 steps).

<table><tr><td colspan="4">Easy
10-34 steps</td><td colspan="4">Medium
35-59 steps</td><td colspan="4">Hard
60-81 steps</td></tr><tr><td>1</td><td>2</td><td>3</td><td>1</td><td>2</td><td>3</td><td>1</td><td>2</td><td>3</td><td>1</td><td>2</td><td>3</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

# J.4.5 Space Invaders

Adapted from the classic arcade game, Space Invaders is a simplified space warfare game. Players control a ship at a grid's bottom, moving it by column to fire lasers upward. Lasers destroy the nearest alien invader in that column (different colors worth 10, 20, or 30 points), earning points and potentially exposing others. Collectively moving enemies add dynamic challenge. The game uses visually intuitive images for ship and aliens instead of text symbols.

Players analyze game scene images to answer questions covering: game state perception (e.g., enemy counts by location or color); single-shot outcome prediction (points from current or post-move shots); effects of consecutive shots in dynamic scenarios; and strategic planning for maximum points. These questions range from simple recognition to complex reasoning. Three difficulty levels are based on scene complexity; higher levels feature larger grids with more numerous and complexly arranged enemies, demanding greater player skill.

<table><tr><td colspan="7">Easy</td><td colspan="10">Medium</td><td colspan="10">Hard</td></tr><tr><td></td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

# J.4.6 Snake

This game is derived from the classic game Snake, which involves a square white grid scene with coordinates, with snakes and food represented by colored squares. The snake head is represented by a yellow square, the snake body by blue squares, and the food by a red square. Each step the snake can move in four directions: up and down, left and right. The game ends if the snake head hits the bound of the grid or its own body.

The questions include 1. The coordinate of the snake head. 2. The coordinate of the food. 3. The length of the snake. 4. Which will happen until this process ends if following a specific sequence of moves (hitting its own body, hitting the wall, reaching the food, or nothing happens)? 5. The length of the shortest path to reach the food. Plot Level is determined by the grid size.

![](images/6552390a7d313bfa7bdb3fde74591d95b88673b5893cc2d3d12eaf5bd0da5716.jpg)

# J.4.7 Chess Ranger

Chess ranger is derived from chess. The game presents a problem with an $8 \times 8$ chessboard image containing 6 pieces, where the possible types of pieces are King, Queen, Rook, Bishop, Knight, and Pawn. The goal of the game is to use the movement and capture rules of chess pieces to ensure that only one piece remains on the board at the end.

The types of questions in the game are as follows:1. The number of pieces of a certain type on the board.2. The identity of the piece located in a specific square on the board.3. The location of a particular type of piece on the board.4. The required number of steps to solve the current chessboard configuration.5. The moves that can solve the puzzle among several possible options. The difficulty level of the game is determined by the number of pieces on the board: 4, 5, 6 pieces corresponding to easy, medium and hard.

![](images/eaaf1259bbeeb187b7101a9f966c7a1c271e81ea579317df4eb0e7df52ecefe4.jpg)

# J.4.8 Pacman

The game is inspired by the classic maze game Pac-Man, with the original four ghosts simplified to just two ghosts. The objective of the game is for Pac-Man to eat as many beans as possible while avoiding being caught by the ghosts. The game scene includes Pac-Man, beans, walls, and ghosts (Pinky and Blinky), with Pac-Man, Pinky, and Blinky represented by special images. The beans are represented as small yellow circles, and the walls are dark blue squares. Pac-Man, Pinky, and Blinky cannot move through walls. The dataset includes tasks such as determining Pac-Man's current position and direction, counting the number of beans in a specific area, predicting the paths of the ghosts, forecasting the outcome of Pac-Man's movements, and analyzing strategies to maximize the score while avoiding ghosts. The dataset is divided into three difficulty levels based on grid size: Easy (16x16), Medium (18x18), and Hard (20x20).

![](images/891d5528fdb4e99fbaf7dde79436da9f2c2c18eb1ab54cb78c10a6c88a21a640.jpg)  
Easy   
$16\times 16$   
Medium   
$18\times 18$   
Hard   
$18\times 18$

![](images/c07f63e9034316d028b04677cf3e2b16802b0a40ba787e681d42e2e53c77342b.jpg)

![](images/dfc9dfa2d7a16cd91dadddbb71cccf67d5567f908ce18eb018051d1ea5ea6ce9.jpg)