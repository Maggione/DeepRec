ripple format is designed for Ripple Network

- Three file is required (for example):
    - ./data/ripple/kg.txt : knowledge graph file
    - ./data/ripple/user_click.txt : user's click history file
    - ./data/ripple/train.txt : user-item pair file

- Introduction
    - The format of the knowledge graph file:
        
        [head] + \t + [tails] + \t + [relations]
        
        One [head] may have relations to more than one tail. The [tails] and [relations] are corresponding one by one.

    - The format of the user's click history file:
        
        [userid] + \t + [click1] + \t + [click2] + ... + \t + [clickn]

        The first term is the [userid] followed by his click history.

    - The format of the user-item pair file
        
        [userid] + \t + [itemid] + \t + [label]

- Note
    - The vocabulary of [head], [click#] and [itemid] is the same.

