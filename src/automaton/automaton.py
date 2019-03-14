from PySimpleAutomata import DFA, NFA, automata_IO

def read_determinize_minimized_write(path, model):

    nfa = automata_IO.nfa_dot_importer(path + model)
    dfa = NFA.nfa_determinization(nfa)
    automata_IO.dfa_to_dot(dfa, model + '_det', path)
    new_dfa = DFA.dfa_minimization(dfa)
    automata_IO.dfa_to_dot(new_dfa, model + "_det_min", path)

def check_relation(nfa1, nfa2):
    '''
    :param nfa1: nfa
    :param nfa2: nfa
    :return: - 1 if model1 includes model 2
             - -1 if model2 includes model 1
             - 0 if the models are equal
             - otherwise
    '''
    dfa1 = NFA.nfa_determinization(nfa1)
    dfa1_comp = DFA.dfa_complementation(dfa1)
    dfa2 = NFA.nfa_determinization(nfa2)
    dfa2_comp = DFA.dfa_complementation(dfa2)
    words_only_in_model2 = DFA.dfa_nonemptiness_check(DFA.dfa_intersection(dfa1_comp, dfa2))
    words_only_in_model1 = DFA.dfa_nonemptiness_check(DFA.dfa_intersection(dfa2_comp, dfa1))

    if words_only_in_model1\
            and not words_only_in_model2:
        return 1
    elif words_only_in_model2\
            and not words_only_in_model1:
        dfa_intersect = DFA.dfa_intersection(dfa1_comp, dfa2)
        dfa_intersect_minimized = DFA.dfa_co_reachable(DFA.dfa_minimization(dfa_intersect))
        automata_IO.dfa_to_dot(dfa_intersect_minimized, 'csv_intersect', path)
        return -1
    elif not words_only_in_model2\
            and not words_only_in_model1:
        return 0
    else:
        return -1


if __name__ == '__main__':


    path = '../../data/logs/example/cvs/ktails_models/'
    # read_determinize_minimized_write(output=path, model='csv_std.dot')
    # check_equivelence(path + "csv_std.dot", path + "csv_std2.dot")
    # check_equivelence(path + "csv_4__model.dot", path + "csv_5__model.dot")
    # check_equivelence(path + "csv_5__model.dot", path + "csv_past_5__model.dot")

    # nfa1 = automata_IO.nfa_dot_importer(path + "csv_3__model.dot")
    # nfa2 = automata_IO.nfa_dot_importer(path + "csv_past_3__model.dot")
    # print('relation:', check_relation(nfa1, nfa2))
    #


    # nfa1 = automata_IO.nfa_dot_importer(path + "csv_10__model.dot")
    # nfa2 = automata_IO.nfa_dot_importer(path + "csv_past_10__model.dot")
    # dfa1 = NFA.nfa_determinization(nfa1)
    # dfa1 = DFA.dfa_minimization(dfa1)
    # dfa2 = NFA.nfa_determinization(nfa2)
    # dfa2 = DFA.dfa_minimization(dfa2)
    # dfa1 = DFA.dfa_co_reachable(dfa1)
    # dfa2 = DFA.dfa_co_reachable(dfa2)
    # automata_IO.dfa_to_dot(dfa1, 'csv_10__model' + "_det_min", path)
    # automata_IO.dfa_to_dot(dfa2, 'csv_10__model_past' + "_det_min", path)
    # print('relation:', check_relation(nfa1, nfa2))

    nfa1 = automata_IO.nfa_dot_importer(path + "csv_7__model.dot")
    nfa2 = automata_IO.nfa_dot_importer(path + "csv_past_7__model.dot")
    nfa3 = automata_IO.nfa_dot_importer(path + "cvs.net.dot")
    print('relation:', check_relation(nfa1, nfa2))
    print('relation:', check_relation(nfa1, nfa3))
    print('relation:', check_relation(nfa2, nfa3))

    # nfa1 = automata_IO.nfa_dot_importer(path + "csv_10__model.dot")
    # nfa2 = automata_IO.nfa_dot_importer(path + "csv_past_10__model.dot")
    # print('relation:', check_relation(nfa1, nfa2))

    # check_equivelence(path + "csv_2__model.dot", path + "csv_past_2__model.dot")
    # check_equivelence(path + "csv_3__model.dot", path + "csv_past_3__model.dot")


