import os
import time
from sys import argv, exit
import string
import json
import errno
from agent import Agent
import numpy as np
import tensorflow as tf
from pathlib import Path as p

fifo_suppression_value = 'fifo_suppression_value' #we don't care if this file is chaged because we write to this file
fifo_object_details = 'fifo_object_details'
num = 0

def generate_positional_embedding(text):
    alphabet = string.ascii_lowercase  # Get all lowercase alphabets
    digits = string.digits  # Get all digits
    symbols = "!@#$%^&*()_-+={}[]|\:;'<>?,./\""

    # Expand the vocabulary to include the extra characters
    vocab = alphabet + digits + symbols

    # Define the desired embedding dimension
    embedding_dimension = 170

    # Create the embedding vector
    embedding = []

    for idx, char in enumerate(text.lower()):
        if char in vocab:
            char_idx = vocab.index(char)
            position_index = idx

            # Add character index and position index to the embedding vector
            embedding.append(char_idx)
            embedding.append(position_index)

    # If the length of the embedding is less than the desired dimension, pad with zeros
    while len(embedding) < embedding_dimension:
        embedding.append(0)

    # Ensure the final embedding has the desired dimension
    embedding = embedding[:embedding_dimension]
    return(embedding)


experience_dict = {}
done = 0
if __name__ == "__main__":
  print ("The reinforcement is starting........", time.time())
  print("*********", p.cwd(), "*************") # for debugging

  try:
    os.mkfifo(fifo_object_details, mode=0o777)
  except OSError as oe:
    if oe.errno != errno.EEXIST:
      print("File does not exist!!!")
      raise

  counter = 1 # this is only for testing, modify accordingly
  while True:
    try:
      read_pipe = os.open(fifo_object_details, os.O_RDONLY)
      bytes = os.read(read_pipe, 1024)
      if len(bytes) == 0:
        print("length of byte is zero")
        break
      print("Read the fifo file - ", counter, " time ", time.time())
      states_string = bytes.decode()
      states_dict = json.loads(states_string)   
      states_values = [str(value) for value in states_dict.values()]
      result = '/'.join(states_values)
      embeddings = generate_positional_embedding(result)
      new_embeddings = tf.convert_to_tensor([embeddings], dtype=tf.float32)
      print(new_embeddings)
      agent = Agent()
      action = agent.choose_action(new_embeddings)

      print("time in millisecond ", time.time(), "Counter", counter, "- Final action  = ", action)
      os.close(read_pipe)

      write_pipe = os.open(fifo_object_details, os.O_WRONLY)
    
      response = "{}".format(action)
      os.write(write_pipe, response.encode())
      os.close(write_pipe)
      print("writing file closed ", time.time())     

      if states_dict["prefix_name"] in experience_dict.keys():
        dc = states_dict["ewma_dc"]
        rtt = float(states_dict["rtt"])
        srtt = float(states_dict["srtt"])
        reward = agent.get_reward(dc, rtt, srtt)
        print("Reward", reward)
        previous_state = experience_dict[states_dict["prefix_name"]] 
        current_state = new_embeddings
          
        start_training = time.time()
        # if counter%100 == 0:
        #   agent.learn(previous_state, reward, current_state,done)
        end_training = time.time()
        print("Training period ", (end_training - start_training)*1000)
      experience_dict[states_dict["prefix_name"]] = new_embeddings
      counter = counter + 1
      print("Counter in RL: ", counter)
    except Exception as e:
      print ("exception: ", e)
  