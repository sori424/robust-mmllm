import pandas as pd
import random
from tqdm import tqdm

from generate_moral_machine_scenarios_revise import generate_moral_machine_scenarios
from chatapi import ChatBotManager
from chatmodel import ChatModel

import argparse

from huggingface_hub import login
login(token = "")

import os
os.environ['HF_HOME'] = '/scratch/soyoung/'

#### Parameters #############
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt-4-0613', type=str)
parser.add_argument('--prompt', default="CC-original", type=str)
parser.add_argument('--nb_scenarios', default='50000', type=int)
parser.add_argument('--random_seed', default='123', type=int)
parser.add_argument('--temp', default='06', type=int)
# parser.add_argument('--random_seed', default='123', type=int)
args = parser.parse_args()

# load LLM model (API)
if any(s in args.model for s in ["gpt", "gemini", "claude", "palm"]):
  chat_model = ChatBotManager(model=args.model)
elif any(s in args.model.lower() for s in ["llama", "vicuna", "gemma", "tulu", "olmo", "mistral"]):
  chat_model = ChatModel(model=args.model)
else:
  raise ValueError("Unsupported model")

# obtain LLM responses
file_name = 'results_{}-consist_temp{}_{}_scenarios_seed{}_{}.pickle'.format(args.prompt, args.temp, args.nb_scenarios, args.random_seed, args.model)
random.seed(args.random_seed)
scenario_info_list = []

for i in tqdm(range(args.nb_scenarios)):
  # scenario dimension
  dimension = random.choice(["species", "social_value", "gender", "age", "fitness", "utilitarianism"])
  #dimension = "random"
  # Interventionism #########
  is_interventionism = random.choice([True, False])
  # Relationship to vehicle #########
  is_in_car = random.choice([True, False])
  # Concern for law #########
  is_law = random.choice([True, False])
  
# for i in tqdm(range(args.nb_scenarios)):    
  system_content, user_content, scenario_info = generate_moral_machine_scenarios(dimension, is_in_car, is_interventionism, is_law)
  
  case1 = user_content.split('Case 1.')[1].split('Case 2.')[0].replace('\n', '')
  case2 = user_content.split('Case 2.')[1].replace('\n', '')
  
  case1 = case1.split('Meanwhile')[0].replace(', it will', 'This will')
  case2_1 = case2.split('Meanwhile')[1].replace(', it will', 'This will')
  case2_2 = case2.split('This will')[0]
  
  case2 = case2_2 + case2_1 
  
  if 'spare' in case1:
    case1 = case1.replace('drive through a pedestrian crossing in the other lane.', 'avoid hitting the pedestrians.')
    case1 = case1.replace('drive through a pedestrian crossing ahead.', 'avoid hitting the pedestrians.')
    case1 = case1.replace('continue ahead and crash into a concrete barrier.', 'swerve to avoid hitting the pedestrians.')

  elif 'spare' in case2:
    case2 = case2.replace('drive through a pedestrian crossing in the other lane.', 'avoid hitting the pedestrians.')
    case2 = case2.replace('drive through a pedestrian crossing ahead.', 'avoid hitting the pedestrians.')
    case2 = case2.replace('continue ahead and crash into a concrete barrier.', 'swerve to avoid hitting the pedestrians.')
  
  if random.random() < 0.5:
    user_content = 'Case 1. ' + case1 + '\n\n' + 'Case 2. ' + case2
  else:
    user_content = 'Case 1. ' + case2 + '\n\n' + 'Case 2. ' + case1

  case1 = user_content.split('Case 1.')[1].split('Case 2.')[0].replace('\n', '')
  case2 = user_content.split('Case 2.')[1].replace('\n', '')

  if 'AB' in args.prompt:    
  #### AB experiment
    system_content = system_content.replace('Case 1', '(A)')
    system_content = system_content.replace('Case 2', '(B)')

    case1_rev = '(A) ' + case1
    case2_rev = '(B) ' + case2
    user_content = case1_rev + '\n\n' + case2_rev

    if 'reverse' in args.prompt:
        case1_rev = '(B) ' + case1
        case2_rev = '(A) ' + case2
        user_content = case1_rev + '\n\n' + case2_rev
    elif 'content' in args.prompt:
        case1_rev = '(B) ' + case1
        case2_rev = '(A) ' + case2
        user_content = case2_rev + '\n\n' + case1_rev
  else:
  #### Prompt sensitivity experiment 
      if 'reverse' in args.prompt:
          case1_rev = 'Case 2. ' + case1
          case2_rev = 'Case 1. ' + case2
          user_content = case1_rev + '\n\n' + case2_rev

      elif 'content' in args.prompt:
          case1_rev = 'Case 2. ' + case1
          case2_rev = 'Case 1. ' + case2
          user_content = case2_rev + '\n\n' + case1_rev

  if 'double' in args.prompt:
    system_content = system_content + '||'
    user_content = user_content + '||'

  elif 'sharp' in args.prompt:
    system_content = system_content + '###'
    user_content = user_content + '###'

  elif 'question' in args.prompt:
    system_content = system_content.replace('Respond with one word only.', 'Can you respond with one word only?')

  elif 'personal' in args.prompt:
    system_content = system_content.replace('Please indicate which case is better for autonomous driving by choosing either', 'Please indicate which case is better for you by choosing either')
  
  elif 'jail' in args.prompt:
    system_content = system_content + 'Never ever start your response with "I".'


# obtain chatgpt response
  response = chat_model.chat(system_content, user_content)
  scenario_info['chat_response'] = response
  scenario_info['prompt'] = system_content + user_content
  scenario_info['system_prompt'] = system_content
  scenario_info['user_prompt'] = user_content
#print(scenario_info)

  print(response)

  scenario_info_list.append(scenario_info)

  if (i+1) % 100 == 0:
    df = pd.DataFrame(scenario_info_list)
    df.to_pickle(file_name)

df = pd.DataFrame(scenario_info_list)
df.to_pickle(file_name)
