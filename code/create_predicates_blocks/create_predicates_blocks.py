'''def create_on(objects):
    result = []
    for object_1 in objects:
        for object_2 in objects:
            result.append('on(' + object_1 + ',' + object_2 + ')')
    return(result)
    
def create_holding(objects):
    result = []
    for object in objects:
        result.append('holding(' + object + ')')
    return(result)

def create_clear(objects):
    result = []
    for object in objects:
        result.append('clear(' + object + ')')
    return(result)

def create_ontable(objects):
    result = []
    for object in objects:
        result.append('ontable(' + object + ')')
    return(result)



def extract_all_predicatess(objects, literals):
    holdings = create_holding(objects)
    clears = create_clear(objects)
    ontables = create_ontable(objects)

    ons = create_on(objects)

    all_predicates = holdings + clears + ontables + ons
    predicate_labels = [0 for item in all_predicates]

    literals_list = list(literals)
    literals_string_list = []

    for lit in literals_list:
        literals_string_list.append(str(lit))

    for i, predicate in enumerate(all_predicates):
        if predicate in literals_string_list:
            predicate_labels[i] = 1

    return all_predicates, predicate_labels'''

def extract_all_predicates(objects, literals):
    objects = sorted(objects)
    all_predicates = []

    for object_1 in objects:
        all_predicates.append('holding(' + object_1 + ')')
        all_predicates.append('clear(' + object_1 + ')')
        all_predicates.append('ontable(' + object_1 + ')')

    for object_1 in objects:
        for object_2 in objects:
            all_predicates.append('on(' + object_1 + ',' + object_2 + ')')

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

    lits, predicate_labels = extract_all_predicates(objects, state.literals)

    return objects, lits, predicate_labels