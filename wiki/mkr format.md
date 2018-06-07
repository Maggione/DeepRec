mkr format is designed for Multi-task learning approach for Knowledge graph enhanced Recommendation.

- Two file is required (for example):
    - ./data/mkr/kg_1_final.txt : knowledge graph file
    - ./data/mkr/ratings_final.txt : user-item pair file

- Introduction
    - The format of the knowledge graph file

        [head] + \t + [tail] + \t + [relation]

        Each line contain a head-tail pair

    - The format of the user-item pair file

        [userid] + \t + [itemid] + \t + [label]

        Each line contain a user-item pair

- Note
    - The vocabulary of [head] and [itemid] is the same.