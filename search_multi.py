from nodes_multi import MultiAircraftNode


class MCTS:
    def __init__(self, node: MultiAircraftNode):
        self.root = node
    
    def tree_policy(self, search_depth):
        curr_node = self.root
        while not curr_node.is_terminal_node(search_depth):
            if curr_node.is_fully_expanded():
                curr_node = curr_node.best_child()
            else:
                return curr_node.expand()        
        return curr_node

    def best_action(self, simulations, search_depth):
        for _ in range(simulations):
            v = self.tree_policy(search_depth)
            reward = v.rollout(search_depth)
            v.backpropagate(reward)
        return self.root.best_child(c_param=0.)

