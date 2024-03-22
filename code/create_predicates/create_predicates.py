def create_move_dir(objects):
    result = []
    for object_1 in objects:
        for object_2 in objects:
            for object_3 in objects:
                result.append('move-dir(' + object_1 + ',' + object_2 + ',' + object_3 + ')')
    return(result)
    
def create_is_nongoal(objects):
    result = []
    for object in objects:
        result.append('is-nongoal(' + object + ')')
    return(result)

def create_clear(objects):
    result = []
    for object in objects:
        result.append('clear(' + object + ')')
    return(result)

def create_is_stone(objects):
    result = []
    for object in objects:
        result.append('is-stone(' + object + ')')
    return(result)

def create_at(objects):
    result = []
    for object_1 in objects:
        for object_2 in objects:
            result.append('at(' + object_1 + ',' + object_2 + ')')
    return result

def create_is_player(objects):
    result = []
    for object in objects:
        result.append('is-player(' + object + ')')
    return(result)

def create_at_goal(objects):
    result = []
    for object in objects:
        result.append('at-goal(' + object + ')')
    return(result)

def create_move(objects):
    result = []
    for object in objects:
        result.append('move(' + object + ')')
    return(result)

def create_is_goal(objects):
    result = []
    for object in objects:
        result.append('is-goal(' + object + ')')
    return(result)


def extract_all_predicatess(objects, literals):
    move_dirs = create_move_dir(objects)
    is_nongoals = create_is_nongoal(objects)
    clears = create_clear(objects)
    is_stones = create_is_stone(objects)
    ats = create_at(objects)
    is_players = create_is_player(objects)
    at_goals = create_at_goal(objects)
    moves = create_move(objects)
    is_goals = create_is_goal(objects)

    all_predicates = move_dirs + is_nongoals + clears + is_stones + ats + is_players + at_goals + moves + is_goals
    predicate_labels = [0 for item in all_predicates]

    literals_list = list(literals)
    literals_string_list = []

    for lit in literals_list:
        literals_string_list.append(str(lit))

    for i, predicate in enumerate(all_predicates):
        if predicate in literals_string_list:
            predicate_labels[i] = 1

    return all_predicates, predicate_labels

def extract_all_predicates(objects, literals):
    move_dirs = []
    is_nongoals = []
    clears = []
    is_stones = []
    ats = []
    is_players = []
    at_goals = []
    moves = []
    is_goals = []

    for object_1 in objects:
        is_nongoals.append('is-nongoal(' + object_1 + ')')
        clears.append('clear(' + object_1 + ')')
        is_stones.append('is-stone(' + object_1 + ')')
        is_players.append('is-player(' + object_1 + ')')
        at_goals.append('at-goal(' + object_1 + ')')
        is_goals.append('is-goal(' + object_1 + ')')
        moves.append('move(' + object_1 + ')')
        for object_2 in objects:
            ats.append('at(' + object_1 + ',' + object_2 + ')')
            for object_3 in objects:
                move_dirs.append('move-dir(' + object_1 + ',' + object_2 + ',' + object_3 + ')')

    all_predicates = move_dirs + is_nongoals + clears + is_stones + ats + is_players + at_goals + moves + is_goals
    predicate_labels = [0 for item in all_predicates]

    literals_list = list(literals)
    literals_string_list = []
    for lit in literals_list:
        literals_string_list.append(str(lit))
    for i, predicate in enumerate(all_predicates):
        if predicate in literals_string_list:
            predicate_labels[i] = 1


    return all_predicates, predicate_labels

def extract_grounded_predicates(env):
    state = env.get_state()

    objects = list(state.objects)

    #print(objects)

    #directions = []
    #locations = []
    #things = []

    #for item in objects:
    #    if 'location' in str(item):
    #        locations.append(item)
    #    elif 'direction' in str(item):
    #        directions.append(item)
    #    elif 'thing' in str(item):
    #        things.append(item)
    #    else:
    #        print("ERROR")

    lits, predicate_labels = extract_all_predicates(objects, state.literals)

    return objects, lits, predicate_labels