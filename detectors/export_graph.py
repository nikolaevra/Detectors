import os
import freeze_graph

# We save out the graph to disk, and then call the const conversion
# routine.
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"

input_graph_path = os.path.join(FLAGS.model_dir, input_graph_name)
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'saved_checkpoint') + "-0"

# Note that we this normally should be only "output_node"!!!
output_node_names = "Dense2/output_node"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = os.path.join(FLAGS.model_dir, output_graph_name)
clear_devices = False

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, input_checkpoint_path,
                          output_node_names, restore_op_name,
                          filename_tensor_name, output_graph_path,
                          clear_devices)